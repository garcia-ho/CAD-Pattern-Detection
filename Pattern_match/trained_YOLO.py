import os
import supervision as sv
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import torch
from collections import defaultdict
from tqdm import tqdm
import csv
from pathlib import Path

def load_model(model_path):
    """
    Load a trained YOLO model from the specified path.
    
    Args:
        model_path (str): Path to the YOLO model file
        
    Returns:
        model: Loaded YOLO model
    """
    print(f"Loading model from {model_path}")
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def segment_image(image, segment_size=1024, overlap=0.15):
    """
    Segment a large image into overlapping patches for processing.
    
    Args:
        image (numpy.ndarray or PIL.Image): Input image to segment
        segment_size (int): Size of each segment
        overlap (float): Overlap between segments as percentage
        
    Returns:
        list: List of (segment, coordinates) tuples where coordinates are (x, y)
    """
    # Convert PIL Image to numpy if needed
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    h, w = image_np.shape[:2]
    
    # Check if image has 4 channels (RGBA) and convert to RGB if needed
    if image_np.shape[-1] == 4:
        print("Converting RGBA image to RGB")
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        h, w = image_np.shape[:2]
    
    # Calculate step size with overlap
    step = int(segment_size * (1 - overlap))
    
    # Calculate number of segments in each dimension
    n_h = max(1, (h - segment_size) // step + 2)
    n_w = max(1, (w - segment_size) // step + 2)
    
    segments = []
    
    for i in range(n_h):
        for j in range(n_w):
            # Calculate segment coordinates
            y = min(i * step, max(0, h - segment_size))
            x = min(j * step, max(0, w - segment_size))
            
            # Handle edge cases
            y_end = min(y + segment_size, h)
            x_end = min(x + segment_size, w)
            
            # Extract segment
            segment = image_np[y:y_end, x:x_end]
            
            # Pad if needed
            if segment.shape[0] < segment_size or segment.shape[1] < segment_size:
                padded = np.zeros((segment_size, segment_size, 3), dtype=np.uint8)
                padded[:segment.shape[0], :segment.shape[1], :] = segment
                segment = padded
            
            # Ensure segment has 3 channels (RGB)
            if segment.shape[-1] != 3:
                segment = cv2.cvtColor(segment, cv2.COLOR_RGBA2RGB)
            
            segments.append((segment, (x, y)))
    
    print(f"Created {len(segments)} segments")
    return segments

def process_segments(model, segments, conf_threshold=0.25, iou_threshold=0.45):
    """
    Process image segments with YOLO model.
    
    Args:
        model: YOLO model
        segments (list): List of (segment, coordinates) tuples
        conf_threshold (float): Confidence threshold
        iou_threshold (float): IoU threshold for NMS
        
    Returns:
        list: List of (detections, coordinates) tuples
    """
    results = []
    
    for segment, coords in tqdm(segments, desc="Processing segments"):
        # Run inference
        try:
            result = model.predict(
                segment, 
                conf=conf_threshold, 
                iou=iou_threshold,
                verbose=False
            )[0]
            
            # Convert to supervision Detections
            detections = sv.Detections.from_ultralytics(result)
            results.append((detections, coords))
        except Exception as e:
            print(f"Error processing segment at {coords}: {e}")
            # Add an empty detection to maintain segment count
            results.append((sv.Detections.empty(), coords))
    
    return results

def merge_detections(segment_results, original_image_size, iou_threshold=0.45):
    """
    Merge detections from segments, avoiding duplicate detections.
    
    Args:
        segment_results (list): List of (detections, coordinates) tuples
        original_image_size (tuple): Original image size (height, width)
        iou_threshold (float): IoU threshold for NMS
        
    Returns:
        sv.Detections: Merged detections
    """
    h, w = original_image_size
    all_boxes = []
    all_confidences = []
    all_class_ids = []
    
    # Convert segment detections to original image coordinates
    for detections, (x_offset, y_offset) in segment_results:
        if len(detections) == 0:
            continue
            
        # Get boxes, converting from segment to original coordinates
        boxes = detections.xyxy
        for i, box in enumerate(boxes):
            # Adjust coordinates
            x1, y1, x2, y2 = box
            x1 += x_offset
            y1 += y_offset
            x2 += x_offset
            y2 += y_offset
            
            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            all_boxes.append([x1, y1, x2, y2])
            all_confidences.append(detections.confidence[i])
            all_class_ids.append(detections.class_id[i])
    
    # Convert to numpy arrays
    if not all_boxes:
        print("No detections found")
        return sv.Detections.empty()
        
    boxes = np.array(all_boxes)
    confidences = np.array(all_confidences)
    class_ids = np.array(all_class_ids)
    
    # Group by class_id for NMS
    indices_by_class = defaultdict(list)
    for i, class_id in enumerate(class_ids):
        indices_by_class[class_id].append(i)
    
    # Perform NMS by class
    keep_indices = []
    for class_id, indices in indices_by_class.items():
        if len(indices) == 0:
            continue
            
        class_boxes = boxes[indices]
        class_confidences = confidences[indices]
        
        # Convert to format for NMS
        boxes_xyxy = torch.tensor(class_boxes, dtype=torch.float32)
        scores = torch.tensor(class_confidences, dtype=torch.float32)
        
        # Apply NMS
        try:
            from torchvision.ops import nms
            keep_idx = nms(boxes_xyxy, scores, iou_threshold)
            keep_indices.extend([indices[i] for i in keep_idx.tolist()])
        except Exception as e:
            print(f"Error in NMS: {e}. Using manual NMS implementation.")
            # Manual NMS implementation
            order = scores.argsort(descending=True).tolist()
            keep = []
            
            while order:
                i = order[0]
                keep.append(i)
                
                if len(order) == 1:
                    break
                    
                # Calculate IoU with the boxes after the current one
                box_i = boxes_xyxy[i]
                other_boxes = boxes_xyxy[order[1:]]
                
                # Calculate intersection coordinates
                xxmin = torch.max(box_i[0], other_boxes[:, 0])
                yymin = torch.max(box_i[1], other_boxes[:, 1])
                xxmax = torch.min(box_i[2], other_boxes[:, 2])
                yymax = torch.min(box_i[3], other_boxes[:, 3])
                
                # Calculate areas
                w = torch.clamp(xxmax - xxmin, min=0)
                h = torch.clamp(yymax - yymin, min=0)
                intersection = w * h
                
                box_area = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
                other_areas = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1])
                
                # Calculate IoU
                union = box_area + other_areas - intersection
                iou = intersection / union
                
                # Keep boxes with IoU less than threshold
                remaining_indices = torch.nonzero(iou < iou_threshold).flatten().tolist()
                order = [order[i+1] for i in remaining_indices]
            
            keep_indices.extend([indices[i] for i in keep])
    
    # Filter boxes, confidences, and class_ids
    if keep_indices:
        boxes = boxes[keep_indices]
        confidences = confidences[keep_indices]
        class_ids = class_ids[keep_indices]
    else:
        # No detections after NMS
        return sv.Detections.empty()
    
    # Create merged Detections object
    return sv.Detections(
        xyxy=boxes,
        confidence=confidences,
        class_id=class_ids
    )

def count_objects(detections, class_names):
    """
    Count occurrences of each class in the detections.
    
    Args:
        detections (sv.Detections): Detection object
        class_names (dict): Dictionary mapping class IDs to names
        
    Returns:
        dict: Dictionary mapping class names to counts
    """
    counts = defaultdict(int)
    
    # Return empty dictionary if no detections
    if len(detections) == 0:
        return {}
        
    for class_id in detections.class_id:
        if class_id in class_names:
            class_name = class_names[class_id]
        else:
            class_name = f"Class {class_id}"
        
        counts[class_name] += 1
    
    return dict(counts)



def annotate_image(image, detections, class_names):
    """
    Annotate an image with detection bounding boxes and labels using supervision.
    
    Args:
        image (numpy.ndarray): Image to annotate (BGR format for OpenCV)
        detections (sv.Detections): Detections object
        class_names (dict): Dictionary mapping class IDs to names
        
    Returns:
        numpy.ndarray: Annotated image
    """
    import supervision as sv
    
    if len(detections) == 0:
        return image
    
    # Make a copy of the image to annotate
    annotated_image = image.copy()
    
    # Check if image is grayscale despite having 3 channels 
    is_effectively_grayscale = False
    if annotated_image.shape[2] == 3:
        sample = annotated_image[::10, ::10]
        if np.all(sample[:,:,0] == sample[:,:,1]) and np.all(sample[:,:,1] == sample[:,:,2]):
            is_effectively_grayscale = True
    
    # Add color markers to corners to verify color processing is working
    if is_effectively_grayscale:
        print("Image appears to be grayscale with 3 identical channels - adding color markers")
        h, w = annotated_image.shape[:2]
        thickness = 20
        annotated_image[0:thickness, 0:thickness] = [0, 0, 255]  # Red square top-left
        annotated_image[0:thickness, w-thickness:w] = [0, 255, 0]  # Green square top-right
        annotated_image[h-thickness:h, 0:thickness] = [255, 0, 0]  # Blue square bottom-left
    
    # Generate a color palette with vibrant colors for each unique class
    unique_classes = set(detections.class_id)
    
    # Define vibrant colors in BGR format (for OpenCV)
    vibrant_colors = [
        (0, 0, 255),      # Red
        (0, 255, 0),      # Green
        (255, 0, 0),      # Blue
        (0, 255, 255),    # Yellow
        (255, 0, 255),    # Magenta
        (255, 255, 0),    # Cyan
        (0, 128, 255),    # Orange
        (255, 0, 128),    # Purple
        (255, 128, 0),    # Light Blue
        (0, 255, 128),    # Lime
        (128, 0, 255),    # Pink
        (255, 128, 255),  # Lavender
        (128, 128, 255),  # Light Red
        (128, 255, 255),  # Light Yellow
        (128, 255, 128)   # Light Green
    ]
    
    # Create a color lookup dictionary
    import random
    random.seed(42)  # For consistent colors between runs
    color_map = {}
    
    for class_id in unique_classes:
        color_map[class_id] = vibrant_colors[class_id % len(vibrant_colors)]
    
    # Manual annotation implementation - always works
    for i, (xyxy, class_id, confidence) in enumerate(zip(
        detections.xyxy, detections.class_id, detections.confidence
    )):
        # Draw box with class-specific color
        x1, y1, x2, y2 = map(int, xyxy)
        
        # Get color for this class - already in BGR for OpenCV
        color = color_map.get(class_id, (0, 255, 0))  # Default to green
        
        # Draw rectangle
        cv2.rectangle(
            annotated_image, 
            (x1, y1), 
            (x2, y2), 
            color, 
            2
        )
        
        # Draw label background
        class_name = class_names[class_id] if class_id in class_names else f"Class {class_id}"
        label = f"{class_name}: {confidence:.2f}"
        
        # Get text size
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        
        # Draw label background
        cv2.rectangle(
            annotated_image,
            (x1, y1 - text_size[1] - 10),
            (x1 + text_size[0] + 10, y1),
            color,
            -1  # Filled
        )
        
        # Draw text (white)
        cv2.putText(
            annotated_image,
            label,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),  # White text
            1
        )
    
    return annotated_image

def process_image(image_path, model, segment_size=1024, overlap=0.15, 
                 conf_threshold=0.25, iou_threshold=0.45):
    """Process a single image with YOLO model."""
    try:
        # Load image
        image = Image.open(image_path)
        
        # Convert to RGB mode if needed (PIL uses RGB, OpenCV uses BGR)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Convert PIL to numpy array
        image_np = np.array(image)
        
        # Convert RGB to BGR for OpenCV processing
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        h, w = image_np.shape[:2]
        
        # Process with YOLO
        segments = segment_image(image_np, segment_size, overlap)
        segment_results = process_segments(model, segments, conf_threshold, iou_threshold)
        merged_detections = merge_detections(segment_results, (h, w), iou_threshold)
        class_names = model.names
        counts = count_objects(merged_detections, class_names)
        
        # Annotate with colorful boxes - pass the BGR image
        annotated_image = annotate_image(image_np, merged_detections, class_names)
        
        # Add a colorful border to confirm color processing is working
        border_size = 30
        annotated_image = cv2.copyMakeBorder(
            annotated_image, 
            border_size, border_size, border_size, border_size,
            cv2.BORDER_CONSTANT, 
            value=[0, 0, 255]  # Red border in BGR
        )
        
        return annotated_image, counts, merged_detections
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        import traceback
        traceback.print_exc()  # Print the full traceback for debugging
        return None, {}, sv.Detections.empty()

def YOLO_process_directory(input_dir, output_dir, model_path, segment_size=1024, 
                     overlap=0.15, conf_threshold=0.25, iou_threshold=0.45):
    """
    Process all images in a directory.
    
    Args:
        input_dir (str): Input directory containing images
        output_dir (str): Output directory for annotated images
        model_path (str): Path to YOLO model
        segment_size (int): Size of segments
        overlap (float): Overlap between segments
        conf_threshold (float): Confidence threshold
        iou_threshold (float): IoU threshold
        
    Returns:
        dict: Dictionary mapping image paths to object counts
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = load_model(model_path)
    
    # Find all images in input directory
    image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(list(Path(input_dir).glob(f'*{ext}')))
    
    print(f"Found {len(image_paths)} images in {input_dir}")
    
    # Process each image
    all_counts = {}
    all_detections = {}
    
    for image_path in tqdm(image_paths, desc="Processing images"):
        # Process image
        annotated_image, counts, detections = process_image(
            str(image_path), 
            model, 
            segment_size, 
            overlap, 
            conf_threshold, 
            iou_threshold
        )
        
        if annotated_image is None:
            continue
        
        # Save annotated image with colors
        output_path = os.path.join(output_dir, image_path.name)
        
        # No need to check for channels or convert grayscale since we've handled that
        # Just convert BGR to RGB for PIL
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        
        # Save with PIL (which expects RGB)
        Image.fromarray(annotated_image_rgb).save(output_path)
        
        print(f"Saved colorful annotated image to {output_path}")
        
        # Store counts
        all_counts[str(image_path)] = counts
        all_detections[str(image_path)] = detections
    
    return all_counts, all_detections

def generate_csv_report(all_counts, output_file="Pattern_match/detection_report.csv"):
    """
    Generate a CSV report with object counts for each image.
    
    Args:
        all_counts (dict): Dictionary mapping image paths to object counts
        output_file (str): Path to output CSV file
        
    Returns:
        str: Path to CSV file
    """
    if not all_counts:
        print("No detections to report")
        return None
    
    # Get all unique class names across all images
    all_class_names = set()
    for counts in all_counts.values():
        all_class_names.update(counts.keys())
    
    # Sort class names for consistent output
    all_class_names = sorted(all_class_names)
    
    # Prepare CSV header
    header = ["Image"] + list(all_class_names)
    
    # Prepare rows
    rows = []
    for image_path, counts in all_counts.items():
        # Use filename without extension as identifier
        image_name = os.path.basename(image_path)
        
        # Create row with counts for each class
        row = [image_name]
        for class_name in all_class_names:
            row.append(counts.get(class_name, 0))
        
        rows.append(row)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Write CSV file
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    
    print(f"CSV report saved to {output_file}")
    return output_file