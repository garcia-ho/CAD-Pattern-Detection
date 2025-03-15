import cv2
import numpy as np
import os
import concurrent.futures
import argparse
from functools import partial
from datetime import datetime

class PatternDetector:
    def __init__(self, threshold=0.7, scale_range=(0.2, 2.0), scale_steps=10, rotations=None):
        """
        Initialize the pattern detector.
        
        Args:
            threshold: Matching threshold (0-1), higher values mean more strict matching
            scale_range: Tuple of (min_scale, max_scale) to search through
            scale_steps: Number of scale steps to use
            rotations: List of rotation angles to check (in degrees), defaults to [0, 90, 180, 270]
        """
        self.threshold = threshold
        self.scale_range = scale_range
        self.scale_steps = scale_steps
        self.rotations = rotations if rotations is not None else [0, 90, 180, 270]
        
    def detect_pattern(self, image, template):
        """
        Detect occurrences of the complete template in the image at multiple scales and rotations.
        
        Args:
            image: Input image to search in
            template: Template image to search for
            
        Returns:
            List of rectangles (x, y, w, h, angle) where pattern was found
        """
        # Convert images to grayscale
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
            
        if len(template.shape) == 3:
            gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            gray_template = template
            
        # Enhance edges to focus on the complete structure
        gray_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
        gray_template = cv2.GaussianBlur(gray_template, (3, 3), 0)
        
        # Get dimensions
        img_h, img_w = gray_image.shape
        
        # Create scale list
        min_scale, max_scale = self.scale_range
        scales = np.linspace(min_scale, max_scale, self.scale_steps)
        
        # Store all matches from different scales and rotations
        all_rectangles = []
        
        # Try different rotations
        for angle in self.rotations:
            # Rotate template
            if angle == 0:
                rotated_template = gray_template
            else:
                # Rotate around center
                center = (gray_template.shape[1] // 2, gray_template.shape[0] // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated_template = cv2.warpAffine(
                    gray_template, 
                    rotation_matrix, 
                    (gray_template.shape[1], gray_template.shape[0]),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=255
                )
            
            # Generate scales for matching
            for scale in scales:
                # Resize template based on scale
                scaled_h = int(rotated_template.shape[0] * scale)
                scaled_w = int(rotated_template.shape[1] * scale)
                
                # Skip scales that would make the template larger than the image
                if scaled_h > img_h or scaled_w > img_w or scaled_h <= 10 or scaled_w <= 10:
                    continue
                    
                # Resize template
                scaled_template = cv2.resize(rotated_template, (scaled_w, scaled_h))
                
                # Perform template matching with normalized correlation coefficient
                # This method is better at finding complete patterns rather than partial ones
                result = cv2.matchTemplate(gray_image, scaled_template, cv2.TM_CCOEFF_NORMED)
                
                # Find locations where the matching exceeds our threshold
                locations = np.where(result >= self.threshold)
                
                # Convert to list of rectangles with rotation angle
                for pt in zip(*locations[::-1]):  # Swap columns and rows
                    all_rectangles.append((pt[0], pt[1], scaled_w, scaled_h, angle))
                    
                # Print status for large-scale operations
                if len(locations[0]) > 0:
                    print(f"Found {len(locations[0])} matches at angle={angle}, scale={scale:.2f}")
        
        # Apply non-maximum suppression with a higher overlap threshold
        filtered_rectangles = self._non_max_suppression(all_rectangles, overlap_threshold=0.3)
            
        return filtered_rectangles
    
    def _non_max_suppression(self, rectangles, overlap_threshold=0.5):
        """
        Apply simplified non-maximum suppression to avoid duplicate detections
        
        Args:
            rectangles: List of rectangles (x, y, w, h, angle)
            overlap_threshold: Maximum allowed overlap (higher means less strict filtering)
            
        Returns:
            Filtered list of rectangles
        """
        if not rectangles:
            return []
            
        # Convert rectangles to format [x1, y1, x2, y2, angle]
        boxes = []
        for x, y, w, h, angle in rectangles:
            boxes.append([x, y, x + w, y + h, angle])
        
        boxes = np.array(boxes)
        
        # Get coordinates
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        angles = boxes[:, 4]
        
        # Compute areas
        areas = (x2 - x1) * (y2 - y1)
        
        # Sort by area (larger first)
        order = np.argsort(areas)[::-1]
        
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
            
            # Keep detections at different angles even if they overlap
            angle_diff = np.abs(angles[i] - angles[order[1:]])
            angle_diff = np.minimum(angle_diff, 360 - angle_diff)
            
            # Only filter out boxes with high IoU AND same angle
            inds = np.where((iou <= overlap_threshold) | (angle_diff >= 45))[0]
            order = order[inds + 1]
            
        # Return filtered rectangles
        return [rectangles[i] for i in keep]
    
    def highlight_matches(self, image, matches, color=(0, 0, 255), thickness=2):
        """
        Highlight the matches in the image.
        
        Args:
            image: Input image
            matches: List of rectangles (x, y, w, h, angle)
            color: Color to use for highlighting (BGR)
            thickness: Line thickness
            
        Returns:
            Image with highlighted matches
        """
        result = image.copy()
        
        for match in matches:
            x, y, w, h, angle = match
            
            if angle % 180 == 0:  # For 0 and 180 degrees
                # Draw rectangle outline
                cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
                
                # Add rotation text
                cv2.putText(result, f"{angle}°", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, color, 1)
            else:
                # For rotated rectangles
                rect = ((x + w/2, y + h/2), (w, h), angle)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                
                # Draw polygon outline
                cv2.drawContours(result, [box], 0, color, thickness)
                
                # Add rotation text
                cv2.putText(result, f"{angle}°", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, color, 1)
        
        return result
        
    def draw_match_count(self, image, count):
        """
        Draw match count on the image.
        
        Args:
            image: Input image
            count: Number of matches
            
        Returns:
            Image with text showing match count
        """
        result = image.copy()
        text = f"Matches: {count}"
        cv2.putText(result, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return result

# Define a parallel foreach function with dopar-like behavior
def foreach_dopar(iterable, func, max_workers=6):
    """
    Execute a function on each item in parallel.
    Similar to R's foreach %dopar% functionality.
    
    Args:
        iterable: Collection to iterate over
        func: Function to apply to each item
        max_workers: Maximum number of parallel workers
    
    Returns:
        List of results
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(func, iterable))
    return results

def process_image(image_name, template, detector, input_dir, output_dir):
    """
    Process a single image with the given detector and template.
    
    Args:
        image_name: Name of the image file
        template: Template image
        detector: PatternDetector instance
        input_dir: Directory containing input images
        output_dir: Directory for output images
    
    Returns:
        Tuple of (image_name, num_matches)
    """
    # Load the test image from input_CAD directory
    image_path = os.path.join(input_dir, image_name)
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not read test image {image_path}")
        return image_name, 0
    
    print(f"\nProcessing {image_name}...")
    print(f"Image dimensions: {image.shape}")
    print(f"Template dimensions: {template.shape}")
    
    # Detect pattern matches at multiple scales and rotations
    matches = detector.detect_pattern(image, template)
    print(f"Found {len(matches)} matches after filtering")
    
    # Highlight matches
    result = detector.highlight_matches(image, matches, color=(0, 0, 255))  # Red color
    
    # Add match count text
    result = detector.draw_match_count(result, len(matches))
    
    # Save the result
    output_path = os.path.join(output_dir, f"processed_{image_name}")
    cv2.imwrite(output_path, result)
    
    print(f"Processed {image_name}: Found {len(matches)} matches of pattern")
    
    return image_name, len(matches)

def generate_detection_report(results, output_file="detection_result.txt"):
    """
    Generate a text report of detection results
    
    Args:
        results: List of (image_name, match_count) tuples
        output_file: Path where to save the report
        
    Returns:
        None
    """
    with open(output_file, 'w') as f:
        f.write("CAD Pattern Detection Results\n")
        f.write("==========================\n\n")
        f.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Summary:\n")
        for image_name, count in results:
            f.write(f"  {image_name}: {count} matches\n")
        
        f.write("\nDetailed Results:\n")
        for image_name, count in results:
            f.write(f"\nImage: {image_name}\n")
            f.write(f"  Number of patterns detected: {count}\n")
            if count == 0:
                f.write("  Status: No patterns detected\n")
            elif count < 2:
                f.write("  Status: Pattern found\n")
            else:
                f.write("  Status: Multiple patterns found\n")
    
    print(f"Detection report saved to {output_file}")

def main():
    """
    Main function to process images and detect patterns
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description='CAD Pattern Detector')
    
    # Detection parameters
    parser.add_argument('--threshold', type=float, default=0.39,
                        help='Matching threshold (0-1), higher values mean stricter matching')
    parser.add_argument('--min-scale', type=float, default=0.1,
                        help='Minimum scale to search')
    parser.add_argument('--max-scale', type=float, default=0.5,
                        help='Maximum scale to search')
    parser.add_argument('--scale-steps', type=int, default=10,
                        help='Number of scale steps between min and max')
    parser.add_argument('--no-rotations', action='store_true',
                        help='Disable rotation detection (only detect at 0 degrees)')
    
    # Processing parameters
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of worker threads for parallel processing')
    parser.add_argument('--report', type=str, default='detection_result.txt',
                        help='Path for the detection report output file')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Determine rotations to use
    rotations = [0] if args.no_rotations else [0, 90, 180, 270]
    print(f"Detecting patterns with rotations: {rotations}")

    # Initialize the pattern detector with parameters from command line
    detector = PatternDetector(
        threshold=args.threshold,
        scale_range=(args.min_scale, args.max_scale),
        scale_steps=args.scale_steps,
        rotations=rotations
    )
    
    # Load the template pattern
    template_path = os.path.join("input_target", "targetA.png")
    template = cv2.imread(template_path)
    
    if template is None:
        print(f"Error: Could not read template image {template_path}")
        return
        
    # Create output directory if it doesn't exist
    output_dir = "output_CAD"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process test images from input_CAD directory
    input_dir = "input_CAD"
    test_images = ["testB.png", "testC.png"]
    
    # Create partial function with fixed parameters
    process_func = partial(
        process_image, 
        template=template, 
        detector=detector, 
        input_dir=input_dir, 
        output_dir=output_dir
    )
    
    # Process images in parallel using foreach_dopar
    results = foreach_dopar(test_images, process_func, max_workers=args.workers)
    
    # Print summary of results
    print("\nSummary of results:")
    for image_name, count in results:
        print(f"  {image_name}: {count} matches")
    
    print("All images processed successfully")
    
    # Generate detection report
    generate_detection_report(results, output_file=args.report)
    
    return results

if __name__ == "__main__":
    main()