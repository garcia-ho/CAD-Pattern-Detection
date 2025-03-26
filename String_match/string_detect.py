#!/usr/bin/env python3
# filepath: /home/r27user6/CAD_matching/String_match/string_detect.py

import os
import re
import sys
import glob
import time
import math
import csv
import argparse
import colorsys
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import pandas as pd
import random
import warnings

# Suppress PaddleOCR warnings about ccache
warnings.filterwarnings("ignore", message="No ccache found")

# Try to import PaddleOCR
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    print("ERROR: PaddleOCR not available. Please install with: pip install paddlepaddle paddleocr")
    sys.exit(1)


def load_target_words(target_file="String_match/target.txt"):
    """
    Load target words from a file
    
    Args:
        target_file (str): Path to file containing target words
        
    Returns:
        list: List of target words
    """
    try:
        with open(target_file, 'r', encoding='utf-8') as f:
            # Read lines, strip whitespace, and filter out empty lines
            words = [line.strip() for line in f.readlines() if line.strip()]
            
        print(f"Loaded {len(words)} target words from {target_file}")
        return words
    except Exception as e:
        print(f"Error loading target words: {e}")
        return []


def generate_distinct_colors(n):
    """
    Generate n visually distinct colors
    
    Args:
        n (int): Number of colors to generate
        
    Returns:
        list: List of RGB tuples (values 0-255)
    """
    colors = []
    
    # Use HSV color space for better distribution
    for i in range(n):
        h = i / n
        s = 0.8 + random.random() * 0.2  # 0.8-1.0
        v = 0.8 + random.random() * 0.2  # 0.8-1.0
        
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    
    # Shuffle colors to ensure adjacent words have distinct colors
    random.shuffle(colors)
    
    return colors


def split_image_into_patches(image, patch_size=1024, overlap_percent=15):
    """
    Split an image into overlapping patches
    
    Args:
        image: NumPy array or path to image file
        patch_size (int): Size of patches (both width and height)
        overlap_percent (int): Percentage of overlap between patches
        
    Returns:
        dict: Dictionary with patches and their coordinates
    """
    # Load image if path is provided
    if isinstance(image, str):
        image = cv2.imread(image)
    
    if image is None:
        print(f"Error: Could not load image")
        return None
    
    # Calculate overlap in pixels
    overlap_pixels = int(patch_size * overlap_percent / 100)
    step_size = patch_size - overlap_pixels
    
    height, width = image.shape[:2]
    
    # Calculate number of patches in each dimension
    num_patches_x = max(1, math.ceil((width - overlap_pixels) / step_size))
    num_patches_y = max(1, math.ceil((height - overlap_pixels) / step_size))
    
    total_patches = num_patches_x * num_patches_y
    
    print(f"Splitting image ({width}x{height}) into {total_patches} patches " +
          f"({num_patches_x}x{num_patches_y}) with {overlap_percent}% overlap")
    
    patches = []
    
    # Extract patches
    for y in range(num_patches_y):
        for x in range(num_patches_x):
            # Calculate patch coordinates
            start_x = x * step_size
            start_y = y * step_size
            
            # For edge patches, adjust to include full patch size if possible
            if x == num_patches_x - 1:
                start_x = max(0, width - patch_size)
            if y == num_patches_y - 1:
                start_y = max(0, height - patch_size)
            
            # Calculate end coordinates
            end_x = min(start_x + patch_size, width)
            end_y = min(start_y + patch_size, height)
            
            # Extract patch
            patch = image[start_y:end_y, start_x:end_x]
            
            # Store patch with coordinates
            patches.append({
                'patch': patch,
                'x': start_x,
                'y': start_y,
                'width': end_x - start_x,
                'height': end_y - start_y
            })
    
    return patches


def detect_text_in_patch(patch_info, ocr_engine, target_words, case_sensitive=False):
    """
    Detect text in a patch and identify target words
    
    Args:
        patch_info (dict): Patch information with image and coordinates
        ocr_engine: PaddleOCR engine
        target_words (list): List of target words to find
        case_sensitive (bool): Whether to use case-sensitive matching
        
    Returns:
        list: List of detected text boxes matching target words
    """
    patch = patch_info['patch']
    patch_x = patch_info['x']
    patch_y = patch_info['y']
    
    # Save patch to a temporary file for OCR
    temp_file = f"temp_patch_{patch_x}_{patch_y}.jpg"
    cv2.imwrite(temp_file, patch)
    
    # Run OCR on the patch
    try:
        result = ocr_engine.ocr(temp_file, cls=True)
    finally:
        # Ensure temp file is removed
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    # Process results
    matches = []
    
    if result and len(result) > 0 and result[0]:
        for line in result[0]:
            box = line[0]  # Bounding box points
            text, confidence = line[1]  # Text and confidence
            
            # Convert target words for comparison
            if not case_sensitive:
                text_for_comparison = text.lower()
                target_words_for_comparison = [w.lower() for w in target_words]
            else:
                text_for_comparison = text
                target_words_for_comparison = target_words
            
            # Check if any target word is in the detected text
            for idx, target in enumerate(target_words_for_comparison):
                # Define matching patterns
                patterns = [
                    f"\\b{re.escape(target)}\\b",  # Whole word
                    f"{re.escape(target)}"         # Substring
                ]
                
                for pattern in patterns:
                    matches_in_text = list(re.finditer(pattern, text_for_comparison))
                    
                    if matches_in_text:
                        for match in matches_in_text:
                            # Calculate location in patch
                            start_pos = match.start()
                            end_pos = match.end()
                            target_length = end_pos - start_pos
                            
                            # Calculate proportional position in the bounding box
                            total_length = len(text_for_comparison)
                            start_ratio = start_pos / total_length if total_length > 0 else 0
                            end_ratio = end_pos / total_length if total_length > 0 else 1
                            
                            # Calculate adjusted box points based on ratios (estimate)
                            # This is an approximation - assumes text is horizontal and uniform
                            x_vals = [float(p[0]) for p in box]  # Ensure coordinates are floats
                            y_vals = [float(p[1]) for p in box]
                            
                            min_x = min(x_vals)
                            max_x = max(x_vals)
                            width = max_x - min_x
                            
                            # Interpolate positions
                            adjusted_min_x = min_x + width * start_ratio
                            adjusted_max_x = min_x + width * end_ratio
                            
                            # Create box points for the matched word
                            word_box = [
                                [adjusted_min_x, min(y_vals)],  # Top-left
                                [adjusted_max_x, min(y_vals)],  # Top-right
                                [adjusted_max_x, max(y_vals)],  # Bottom-right
                                [adjusted_min_x, max(y_vals)]   # Bottom-left
                            ]
                            
                            # Adjust coordinates to original image
                            global_box = [[float(p[0] + patch_x), float(p[1] + patch_y)] for p in word_box]
                            
                            # Store the match
                            match_info = {
                                'text': text,
                                'target': target_words[idx],
                                'confidence': float(confidence),  # Ensure confidence is float
                                'box': global_box,
                                'pattern_type': 'word' if pattern.startswith('\\b') else 'substring'
                            }
                            
                            matches.append(match_info)
    
    return matches


def is_duplicate_box(new_match, existing_matches, iou_threshold=0.5):
    """
    Check if a box is a duplicate of any existing box
    
    Args:
        new_match: The match info to check
        existing_matches: List of existing matches
        iou_threshold: IoU threshold for considering duplicates
        
    Returns:
        bool: True if duplicate, False otherwise
    """
    # Convert box format for IoU calculation
    def convert_to_rect(box):
        # Extract min/max coordinates
        x_vals = [float(p[0]) for p in box]  # Ensure all values are floats
        y_vals = [float(p[1]) for p in box]
        return [min(x_vals), min(y_vals), max(x_vals), max(y_vals)]
    
    new_box = new_match['box']
    new_target = new_match['target']
    new_rect = convert_to_rect(new_box)
    
    for existing_match in existing_matches:
        # Only compare matches for the same target word
        if existing_match['target'] != new_target:
            continue
            
        try:
            existing_box = existing_match['box']
            existing_rect = convert_to_rect(existing_box)
            
            # Calculate IoU
            x1 = max(new_rect[0], existing_rect[0])
            y1 = max(new_rect[1], existing_rect[1])
            x2 = min(new_rect[2], existing_rect[2])
            y2 = min(new_rect[3], existing_rect[3])
            
            # Check if boxes overlap
            if x1 < x2 and y1 < y2:
                intersection = (x2 - x1) * (y2 - y1)
                area1 = (new_rect[2] - new_rect[0]) * (new_rect[3] - new_rect[1])
                area2 = (existing_rect[2] - existing_rect[0]) * (existing_rect[3] - existing_rect[1])
                union = area1 + area2 - intersection
                iou = intersection / union if union > 0 else 0
                
                if iou > iou_threshold:
                    # Keep the one with higher confidence
                    if existing_match['confidence'] < new_match['confidence']:
                        existing_match.update({
                            'box': new_match['box'], 
                            'confidence': new_match['confidence'],
                            'text': new_match['text']
                        })
                    return True
        except (TypeError, ValueError) as e:
            print(f"Warning: Error comparing boxes: {e}")
            continue
    
    return False


def highlight_text_in_image(image, matches, color_map=None, line_thickness=2,
                          show_confidence=True, font_scale=0.5):
    """
    Highlight detected text in the image
    
    Args:
        image: Image to highlight text in
        matches: List of matched text boxes
        color_map: Dictionary mapping target words to colors
        line_thickness: Thickness of bounding box lines
        show_confidence: Whether to show confidence scores
        font_scale: Scale factor for text
        
    Returns:
        image: Image with highlighted text
    """
    # Make a copy of the image
    result_img = image.copy()
    
    # Default color if no color map is provided
    default_color = (0, 255, 0)  # Green
    
    # Process each match
    for match in matches:
        target = match['target']
        box = match['box']
        confidence = match['confidence']
        
        # Get color for this target
        if color_map and target in color_map:
            color = color_map[target]
        else:
            color = default_color
        
        try:
            # Convert box to numpy array with integer coordinates
            box_np = np.array([[int(p[0]), int(p[1])] for p in box], np.int32)
            
            # Draw polygon around the text
            cv2.polylines(result_img, [box_np], True, color, line_thickness)
            
            # Show confidence if requested
            if show_confidence:
                # Position for text
                text_x = int(box_np[0][0])
                text_y = int(box_np[0][1]) - 5
                
                # Ensure text is within image bounds
                if text_y < 10:
                    text_y = int(box_np[3][1] + 15)
                
                # Draw confidence score
                cv2.putText(result_img, f"{target}: {confidence:.2f}", 
                          (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                          font_scale, color, 2)
        except Exception as e:
            print(f"Warning: Error highlighting text: {e}")
            continue
    
    return result_img


def process_image(image_path, output_path, target_words, ocr_engine,
                patch_size=1024, overlap_percent=15, case_sensitive=False):
    """
    Process a single image - detect text in patches and highlight matches
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the highlighted image
        target_words: List of target words to find
        ocr_engine: PaddleOCR engine
        patch_size: Size of image patches
        overlap_percent: Percentage of overlap between patches
        case_sensitive: Whether to use case-sensitive matching
        
    Returns:
        dict: Dictionary with processing results
    """
    start_time = time.time()
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return {'success': False, 'error': 'Could not load image'}
    
    # Generate colors for each target word
    colors = generate_distinct_colors(len(target_words))
    color_map = {target: colors[i] for i, target in enumerate(target_words)}
    
    # Split image into patches
    patches = split_image_into_patches(image, patch_size, overlap_percent)
    if not patches:
        print(f"Error: Failed to split image {image_path}")
        return {'success': False, 'error': 'Failed to split image'}
    
    # Process each patch
    all_matches = []
    
    for i, patch_info in enumerate(tqdm(patches, desc="Processing patches")):
        try:
            patch_matches = detect_text_in_patch(patch_info, ocr_engine, target_words, case_sensitive)
            
            # Add non-duplicate matches to the list
            for match in patch_matches:
                if not is_duplicate_box(match, all_matches):
                    all_matches.append(match)
        except Exception as e:
            print(f"Warning: Error processing patch {i}: {e}")
            continue
    
    # Count occurrences of each target word
    word_counts = {target: 0 for target in target_words}
    for match in all_matches:
        target = match['target']
        if target in word_counts:
            word_counts[target] += 1
    
    # Highlight text in the image
    highlighted_image = highlight_text_in_image(image, all_matches, color_map)
    
    # Save the highlighted image
    cv2.imwrite(output_path, highlighted_image)
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    return {
        'success': True,
        'image_path': image_path,
        'output_path': output_path,
        'processing_time': processing_time,
        'word_counts': word_counts,
        'total_matches': len(all_matches)
    }


def process_all_images(input_dir="String_match/filtered_img", output_dir="String_match/output", 
                      target_file="String_match/target.txt", patch_size=1024, overlap_percent=15,
                      case_sensitive=False):
    """
    Process all images in a directory
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save output images
        target_file: Path to file containing target words
        patch_size: Size of image patches
        overlap_percent: Percentage of overlap between patches
        case_sensitive: Whether to use case-sensitive matching
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if PaddleOCR is available
    if not PADDLE_AVAILABLE:
        print("ERROR: PaddleOCR is not available. Please install with: pip install paddlepaddle paddleocr")
        return
    
    # Load target words
    target_words = load_target_words(target_file)
    if not target_words:
        print("No target words found. Please check your target file.")
        return
    
    # Initialize PaddleOCR
    print("Initializing PaddleOCR...")
    try:
        ocr_engine = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False, show_log=False)
    except Exception as e:
        print(f"Error initializing PaddleOCR: {e}")
        return
    
    # Find all images in input directory
    image_files = glob.glob(os.path.join(input_dir, "*.jpg")) + \
                  glob.glob(os.path.join(input_dir, "*.png")) + \
                  glob.glob(os.path.join(input_dir, "*.jpeg"))
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} image files to process")
    
    # Prepare CSV file for results
    csv_path = os.path.join(output_dir, "word_counts.csv")
    
    # Create DataFrame to store results
    results_data = []
    
    # Process each image
    for img_idx, img_path in enumerate(image_files, 1):
        img_basename = os.path.basename(img_path)
        output_path = os.path.join(output_dir, f"highlighted_{img_basename}")
        
        print(f"Processing image {img_idx}/{len(image_files)}: {img_basename}")
        
        try:
            # Process the image
            result = process_image(
                img_path, output_path, target_words, ocr_engine,
                patch_size, overlap_percent, case_sensitive
            )
            
            if result['success']:
                processing_time = result['processing_time']
                word_counts = result['word_counts']
                total_matches = result['total_matches']
                
                print(f"Found {total_matches} matches in {processing_time:.2f} seconds")
                
                # Add to results data
                result_row = {'Image': img_basename}
                result_row.update(word_counts)
                result_row['Total'] = total_matches
                result_row['Processing Time (s)'] = round(processing_time, 2)
                
                results_data.append(result_row)
            else:
                print(f"Error processing {img_basename}: {result.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"Error processing {img_basename}: {e}")
    
    # Create DataFrame and save to CSV
    if results_data:
        try:
            df = pd.DataFrame(results_data)
            df.to_csv(csv_path, index=False)
            print(f"Results saved to {csv_path}")
        except Exception as e:
            print(f"Error saving results to CSV: {e}")
    else:
        print("No results to save")
    
    print("All images processed successfully")
