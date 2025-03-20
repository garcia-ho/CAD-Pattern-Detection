import cv2
import numpy as np
import os
import concurrent.futures
import argparse
from functools import partial
from datetime import datetime
import glob
from pathlib import Path

class MultiPatternDetector:
    def __init__(self, threshold=0.7, scale_range=(0.2, 2.0), scale_steps=10, rotations=None, 
                 overlap_threshold=0.3, min_separation=10):
        """
        Initialize the multi-pattern detector.
        
        Args:
            threshold: Matching threshold (0-1), higher values mean more strict matching
            scale_range: Tuple of (min_scale, max_scale) to search through
            scale_steps: Number of scale steps to use
            rotations: List of rotation angles to check (in degrees), defaults to [0, 90, 180, 270]
            overlap_threshold: Maximum allowed overlap between matched regions
            min_separation: Minimum pixel distance between separate matches
        """
        self.threshold = threshold
        self.scale_range = scale_range
        self.scale_steps = scale_steps
        self.rotations = rotations if rotations is not None else [0, 90, 180, 270]
        self.overlap_threshold = overlap_threshold
        self.min_separation = min_separation
        
    def preprocess_image(self, image):
        """
        Preprocess image to optimize for black pattern detection.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed grayscale and edge images
        """
        if len(image.shape) == 3:
            # Filter out all pixels lighter than RGB(128,128,128)
            dark_mask = np.logical_and.reduce((
                image[:,:,0] < 128,
                image[:,:,1] < 128,
                image[:,:,2] < 128
            ))
            
            # Create a white background image
            filtered_image = np.ones_like(image) * 255
            
            # Copy only the dark pixels to the filtered image
            filtered_image[dark_mask] = [0, 0, 0]  # Make all dark pixels pure black
            
            # Convert to grayscale
            gray_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
        else:
            # If already grayscale, filter pixels > 128 (make them white)
            gray_image = image.copy()
            gray_image[gray_image > 128] = 255
            gray_image[gray_image <= 128] = 0  # Make dark pixels pure black
        
        # Threshold to binary image
        _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
        
        # Create edge image
        edge_image = cv2.Canny(binary_image, 50, 150)
        
        # Dilate edges slightly to connect nearby lines
        kernel = np.ones((2,2), np.uint8)
        edge_image = cv2.dilate(edge_image, kernel, iterations=1)
        
        return binary_image, edge_image
    
    def preprocess_template(self, template):
        """
        Preprocess template for pattern matching.
        
        Args:
            template: Template image
            
        Returns:
            Preprocessed binary and edge template images
        """
        return self.preprocess_image(template)
    
    def detect_patterns(self, image, templates):
        """
        Detect all templates in an image concurrently, choosing the best match at each position.
        
        Args:
            image: Input image to search in
            templates: List of (template_image, template_id) pairs
            
        Returns:
            List of detections (template_id, x, y, w, h, angle, score)
        """
        # Preprocess the input image
        binary_image, edge_image = self.preprocess_image(image)
        img_h, img_w = binary_image.shape
        
        # Preprocess all templates
        preprocessed_templates = []
        for template, template_id in templates:
            binary_template, edge_template = self.preprocess_template(template)
            preprocessed_templates.append((binary_template, edge_template, template_id))
        
        # Store all matches from different scales, rotations, and templates
        all_detections = []
        
        # Try different rotations
        for angle in self.rotations:
            rotated_templates = []
            
            # Rotate all templates for this angle
            for binary_template, edge_template, template_id in preprocessed_templates:
                if angle == 0:
                    rotated_binary = binary_template
                    rotated_edges = edge_template
                else:
                    # Rotate around center
                    center = (binary_template.shape[1] // 2, binary_template.shape[0] // 2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                    
                    # Rotate both the binary template and edge template
                    rotated_binary = cv2.warpAffine(
                        binary_template, 
                        rotation_matrix, 
                        (binary_template.shape[1], binary_template.shape[0]),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=255
                    )
                    
                    rotated_edges = cv2.warpAffine(
                        edge_template, 
                        rotation_matrix, 
                        (edge_template.shape[1], edge_template.shape[0]),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0
                    )
                
                rotated_templates.append((rotated_binary, rotated_edges, template_id))
            
            # Generate scales for matching
            min_scale, max_scale = self.scale_range
            scales = np.linspace(min_scale, max_scale, self.scale_steps)
            
            for scale in scales:
                # For each template at this scale and rotation
                scaled_templates = []
                
                for rotated_binary, rotated_edges, template_id in rotated_templates:
                    # Resize template based on scale
                    scaled_h = int(rotated_binary.shape[0] * scale)
                    scaled_w = int(rotated_binary.shape[1] * scale)
                    
                    # Skip scales that would make the template larger than the image
                    if scaled_h > img_h or scaled_w > img_w or scaled_h <= 10 or scaled_w <= 10:
                        continue
                        
                    # Resize both binary and edge templates
                    scaled_binary = cv2.resize(rotated_binary, (scaled_w, scaled_h))
                    scaled_edges = cv2.resize(rotated_edges, (scaled_w, scaled_h))
                    
                    scaled_templates.append((scaled_binary, scaled_edges, template_id, scaled_w, scaled_h))
                
                if not scaled_templates:
                    continue
                
                # Process each template separately instead of trying to combine results
                # This avoids the broadcasting issues caused by different template sizes
                for scaled_binary, scaled_edges, template_id, scaled_w, scaled_h in scaled_templates:
                    # Calculate result matrix size for this template
                    result_h = img_h - scaled_h + 1
                    result_w = img_w - scaled_w + 1
                    
                    # Skip if dimensions are invalid
                    if result_h <= 0 or result_w <= 0:
                        continue
                    
                    # Match with binary and edge templates
                    binary_result = cv2.matchTemplate(binary_image, scaled_binary, cv2.TM_CCOEFF_NORMED)
                    edge_result = cv2.matchTemplate(edge_image, scaled_edges, cv2.TM_CCOEFF_NORMED)
                    
                    # Combine results (weighted average favoring edge matching)
                    combined_result = binary_result * 0.6 + edge_result * 0.4
                    
                    # Find local maxima for this template
                    local_maxima = self._find_local_maxima(combined_result, 
                                                         self.threshold, 
                                                         min_distance=self.min_separation)
                    
                    # Convert to list of detections
                    for pt in local_maxima:
                        x, y = pt
                        score = combined_result[y, x]
                        all_detections.append((template_id, x, y, scaled_w, scaled_h, angle, score))
                    
                    # Print status update for large operations
                    if local_maxima:
                        print(f"Found {len(local_maxima)} matches at angle={angle}, scale={scale:.2f}, template={template_id}")
            
            # Apply non-maximum suppression to remove duplicates, grouping by template_id
            filtered_detections = self._non_max_suppression_by_template(all_detections, 
                                                                      overlap_threshold=self.overlap_threshold)
        
        return filtered_detections
    
    def _find_local_maxima(self, result_matrix, threshold, min_distance=10):
        """
        Find local maxima in the matching result matrix that are above the threshold
        and separated by at least min_distance pixels
        
        Args:
            result_matrix: 2D matching result matrix
            threshold: Minimum value to consider as a match
            min_distance: Minimum distance between peaks in pixels
            
        Returns:
            List of (x, y) coordinates of local maxima
        """
        # Apply threshold
        data = result_matrix.copy()
        data[data < threshold] = 0
        
        # Find local maxima
        coordinates = []
        while True:
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(data)
            if max_val <= 0:
                break
                
            # Add to coordinates
            coordinates.append(max_loc)
            
            # Suppress this peak and its surroundings
            y, x = max_loc[::-1]  # Convert to row, col format
            y_start = max(0, y - min_distance)
            y_end = min(data.shape[0], y + min_distance + 1)
            x_start = max(0, x - min_distance)
            x_end = min(data.shape[1], x + min_distance + 1)
            
            data[y_start:y_end, x_start:x_end] = 0
            
        return coordinates
    
    def _non_max_suppression_by_template(self, detections, overlap_threshold=0.3):
        """
        Apply non-maximum suppression to detections, grouped by template_id
        
        Args:
            detections: List of (template_id, x, y, w, h, angle, score)
            overlap_threshold: Maximum allowed overlap
            
        Returns:
            Filtered list of detections
        """
        if not detections:
            return []
        
        # Group detections by template_id
        detections_by_template = {}
        for detection in detections:
            template_id = detection[0]
            if template_id not in detections_by_template:
                detections_by_template[template_id] = []
            detections_by_template[template_id].append(detection)
        
        # Process each template group separately
        filtered_detections = []
        for template_id, template_detections in detections_by_template.items():
            # Convert to format expected by NMS function
            boxes = []
            for i, (tid, x, y, w, h, angle, score) in enumerate(template_detections):
                boxes.append([x, y, x + w, y + h, angle, score])
            
            if not boxes:
                continue
                
            boxes = np.array(boxes)
            
            # Get coordinates
            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = boxes[:, 2]
            y2 = boxes[:, 3]
            angles = boxes[:, 4]
            scores = boxes[:, 5]
            
            # Compute areas
            areas = (x2 - x1) * (y2 - y1)
            
            # Sort by score
            order = np.argsort(-scores)  # Negative scores for descending order
            
            keep = []
            while order.size > 0:
                i = order[0]
                keep.append(i)
                
                # Exit if this is the last box
                if order.size == 1:
                    break
                    
                # Compute IoU with remaining boxes
                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])
                
                # Width and height of intersection
                w = np.maximum(0, xx2 - xx1)
                h = np.maximum(0, yy2 - yy1)
                
                # IoU
                intersection = w * h
                union = areas[i] + areas[order[1:]] - intersection
                iou = intersection / (union + 1e-6)
                
                # Check center distance for better handling of closely placed patterns
                center_i = ((x1[i] + x2[i]) / 2, (y1[i] + y2[i]) / 2)
                centers = ((x1[order[1:]] + x2[order[1:]]) / 2, (y1[order[1:]] + y2[order[1:]]) / 2)
                
                # Calculate squared distances (avoid sqrt for performance)
                dx = center_i[0] - centers[0]
                dy = center_i[1] - centers[1]
                distances_sq = dx*dx + dy*dy
                
                # Keep detections at different angles even if they overlap
                angle_diff = np.abs(angles[i] - angles[order[1:]])
                angle_diff = np.minimum(angle_diff, 360 - angle_diff)
                
                # Only filter out boxes with high IoU AND same angle AND centers are close
                min_dist_sq = self.min_separation * self.min_separation
                inds = np.where((iou <= overlap_threshold) | 
                               (angle_diff >= 45) | 
                               (distances_sq >= min_dist_sq))[0]
                
                order = order[inds + 1]
            
            # Add kept detections to filtered list
            for i in keep:
                filtered_detections.append(template_detections[i])
        
        return filtered_detections
    
    def highlight_matches(self, image, detections, alpha=0.4):
        """
        Highlight the matches in the image with semi-transparent colored overlays.
        
        Args:
            image: Input image
            detections: List of (template_id, x, y, w, h, angle, score)
            alpha: Transparency level (0-1), where 1 is opaque
            
        Returns:
            Image with highlighted matches
        """
        result = image.copy()
        
        # Generate distinct colors for each template
        unique_template_ids = set(d[0] for d in detections)
        colors = self._generate_distinct_colors(len(unique_template_ids))
        template_colors = {tid: colors[i] for i, tid in enumerate(unique_template_ids)}
        
        # Create an overlay for semi-transparency
        overlay = result.copy()
        
        for detection in detections:
            template_id, x, y, w, h, angle, score = detection
            
            # Get color for this template
            bgr_color = tuple(int(c * 255) for c in template_colors[template_id])
            
            if angle % 180 == 0:  # For 0 and 180 degrees
                # Draw filled rectangle on overlay
                cv2.rectangle(overlay, (x, y), (x + w, y + h), bgr_color, -1)  # Filled
                
                # Draw rectangle outline on result
                cv2.rectangle(result, (x, y), (x + w, y + h), bgr_color, 2)
                
                # Add template ID and confidence
                label = f"ID:{template_id} ({score:.2f})"
                cv2.putText(result, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, bgr_color, 2)
            else:
                # For rotated rectangles
                rect = ((x + w/2, y + h/2), (w, h), angle)
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                
                # Draw filled polygon on overlay
                cv2.fillPoly(overlay, [box], bgr_color)
                
                # Draw polygon outline on result
                cv2.drawContours(result, [box], 0, bgr_color, 2)
                
                # Add template ID and confidence
                label = f"ID:{template_id} ({score:.2f})"
                cv2.putText(result, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, bgr_color, 2)
        
        # Blend overlay with original image for transparency
        cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
        
        return result
    
    def _generate_distinct_colors(self, n):
        """
        Generate n visually distinct colors in RGB format (0-1 range).
        
        Args:
            n: Number of colors to generate
            
        Returns:
            List of RGB color tuples
        """
        import colorsys
        
        # Use HSV space for generating distinct colors
        colors = []
        for i in range(n):
            # Evenly spaced hues
            hue = i / n
            # Full saturation and value for bright colors
            saturation = 0.9
            value = 0.9
            # Convert to RGB
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append(rgb)
        
        return colors
    
def load_templates(template_dir):
    """
    Load all template images from a directory.
    
    Args:
        template_dir: Directory containing template images
        
    Returns:
        List of (template_image, template_id) pairs
    """
    templates = []
    template_files = []
    
    # Support multiple image formats
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']:
        template_files.extend(glob.glob(os.path.join(template_dir, ext)))
    
    # Sort template files for consistent IDs
    template_files.sort()
    
    for i, file_path in enumerate(template_files):
        template = cv2.imread(file_path)
        if template is None:
            print(f"Warning: Could not load template {file_path}")
            continue
        
        # Use filename without extension as ID if possible
        try:
            template_id = int(os.path.splitext(os.path.basename(file_path))[0])
        except ValueError:
            template_id = i + 1  # 1-based IDs
            
        templates.append((template, template_id))
        print(f"Loaded template ID {template_id}: {file_path}")
    
    return templates

def generate_detection_report(detections, image_path, output_file="Pattern_match/detection_result.txt"):
    """
    Generate a report of detected patterns.
    
    Args:
        detections: List of (template_id, x, y, w, h, angle, score)
        image_path: Path to the input image
        output_file: Path to the output report file
    """
    with open(output_file, 'w') as f:
        f.write(f"Detection Report - {datetime.now()}\n")
        f.write(f"Image: {image_path}\n")
        f.write(f"Number of detections: {len(detections)}\n\n")
        
        # Group by template ID
        detections_by_template = {}
        for detection in detections:
            template_id = detection[0]
            if template_id not in detections_by_template:
                detections_by_template[template_id] = []
            detections_by_template[template_id].append(detection)
        
        # Report for each template
        for template_id, template_detections in sorted(detections_by_template.items()):
            f.write(f"Template ID {template_id}: {len(template_detections)} instances\n")
            for i, (tid, x, y, w, h, angle, score) in enumerate(template_detections):
                f.write(f"  {i+1}. Position: ({x}, {y}), Size: {w}x{h}, Angle: {angle}°, Confidence: {score:.3f}\n")
            f.write("\n")
    
    print(f"Detection report saved to {output_file}")

def process_image(image_path, template_dir, output_dir, detector, args):
    """
    Process a single image with all templates.
    
    Args:
        image_path: Path to the input image
        template_dir: Directory containing template images
        output_dir: Directory to save results
        detector: MultiPatternDetector instance
        args: Command line arguments
        
    Returns:
        Tuple of (detections, visualization)
    """
    print(f"Processing image: {image_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None, None
    
    # Load templates
    templates = load_templates(template_dir)
    if not templates:
        print(f"Error: No templates found in {template_dir}")
        return None, None
    
    print(f"Loaded {len(templates)} templates")
    
    # Detect patterns
    start_time = datetime.now()
    detections = detector.detect_patterns(image, templates)
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print(f"Found {len(detections)} pattern instances in {elapsed:.2f} seconds")
    
    # Generate visualization
    visualization = detector.highlight_matches(image, detections)
    
    # Draw summary information
    summary_text = f"Found {len(detections)} instances of {len(set(d[0] for d in detections))} patterns"
    cv2.putText(visualization, summary_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 
               1, (0, 0, 0), 4)  # Black outline
    cv2.putText(visualization, summary_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 
               1, (255, 255, 255), 2)  # White text
    
    # Add time information
    time_text = f"Processing time: {elapsed:.2f} seconds"
    cv2.putText(visualization, time_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 
               0.7, (0, 0, 0), 3)  # Black outline
    cv2.putText(visualization, time_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 
               0.7, (255, 255, 255), 1)  # White text
    
    # Save visualization
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{image_name}_detected.png")
    cv2.imwrite(output_path, visualization)
    print(f"Saved visualization to {output_path}")
    
    # Generate report
    report_path = os.path.join(output_dir, f"{image_name}_report.txt")
    generate_detection_report(detections, image_path, report_path)
    
    return detections, visualization

def main():
    """
    Main function to process images and detect patterns
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Multi-Pattern CAD Detector')
    
    # Input and output options
    parser.add_argument('-i', '--image', type=str, required=False,
                        help='Input image file or directory')
    parser.add_argument('-t', '--templates', type=str, required=False, default='Pattern_match/input_target',
                        help='Directory containing template images')
    parser.add_argument('-o', '--output', type=str, required=False, default='Pattern_match/output_CAD',
                        help='Directory to save results')
    
    # Detection parameters
    parser.add_argument('--threshold', type=float, default=0.55,
                        help='Matching threshold (0-1), higher values mean stricter matching')
    parser.add_argument('--min-scale', type=float, default=0.01,
                        help='Minimum scale to search')
    parser.add_argument('--max-scale', type=float, default=0.5,
                        help='Maximum scale to search')
    parser.add_argument('--scale-steps', type=int, default=20,
                        help='Number of scale steps between min and max')
    parser.add_argument('--no-rotations', action='store_true',
                        help='Disable rotation detection (only detect at 0 degrees)')
    parser.add_argument('--overlap', type=float, default=0.7,
                        help='Maximum overlap threshold for nearby patterns (0-1)')
    parser.add_argument('--min-separation', type=int, default=10,
                        help='Minimum separation between distinct matches in pixels')
    parser.add_argument('--alpha', type=float, default=0.3,
                        help='Transparency level for pattern highlighting (0-1)')
    
    # Processing parameters
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of worker threads for parallel processing')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set default image path if not provided
    if not args.image:
        args.image = 'input'
        print(f"No input specified, using default: {args.image}")
    
    # Determine rotations to use
    rotations = [0] if args.no_rotations else [0, 90, 180, 270]
    print(f"Detecting patterns with rotations: {rotations}")
    
    # Initialize the pattern detector with parameters from command line
    detector = MultiPatternDetector(
        threshold=args.threshold,
        scale_range=(args.min_scale, args.max_scale),
        scale_steps=args.scale_steps,
        rotations=rotations,
        overlap_threshold=args.overlap,
        min_separation=args.min_separation
    )
    
    # Process single image or directory
    if os.path.isfile(args.image):
        # Process single image
        process_image(args.image, args.templates, args.output, detector, args)
    elif os.path.isdir(args.image):
        # Process all images in directory
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']:
            image_files.extend(glob.glob(os.path.join(args.image, ext)))
        
        print(f"Found {len(image_files)} images to process")
        
        if not image_files:
            print("No images found!")
            return
        
        # Use parallel processing if workers > 1
        if args.workers > 1 and len(image_files) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                process_func = partial(process_image, template_dir=args.templates, 
                                      output_dir=args.output, detector=detector, args=args)
                futures = {executor.submit(process_func, image_path): image_path 
                          for image_path in image_files}
                
                for future in concurrent.futures.as_completed(futures):
                    image_path = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")
        else:
            # Process sequentially
            for image_path in image_files:
                try:
                    process_image(image_path, args.templates, args.output, detector, args)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
    else:
        print(f"Error: {args.image} is not a valid file or directory")

if __name__ == "__main__":
    main()