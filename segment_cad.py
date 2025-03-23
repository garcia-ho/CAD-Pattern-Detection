#!/usr/bin/env python3
import os
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

def filter_to_binary(image):
    """
    Filter the image to keep only dark pixels (RGB < 128) as black (0,0,0),
    and make all other pixels white (255,255,255).
    
    Args:
        image: Input BGR image
        
    Returns:
        Binary image with black elements on white background
    """
    # Create a mask for pixels where all RGB values are < 128
    mask = np.all(image < 128, axis=2)
    
    # Create a binary image (black and white)
    binary = np.ones_like(image) * 255  # White background
    binary[mask] = 0  # Black elements
    
    return binary

def segment_large_cad(input_path, output_dir, segment_size=1024, overlap=0.3, min_content=0.02):
    """
    Segment large CAD drawing into overlapping tiles after filtering to binary.
    
    Args:
        input_path: Path to large CAD image
        output_dir: Directory to save segments
        segment_size: Size of each segment (square)
        overlap: Percentage of overlap between segments
        min_content: Minimum content (non-white) percentage to keep a segment
        
    Returns:
        Number of segments created
    """
    # Load image
    img = cv2.imread(str(input_path))
    if img is None:
        print(f"Could not load {input_path}")
        return 0
    
    # Filter to binary (black and white)
    img = filter_to_binary(img)
    
    h, w = img.shape[:2]
    
    # Calculate step size with overlap
    step = int(segment_size * (1 - overlap))
    
    # Calculate number of segments
    n_h = max(1, (h - segment_size) // step + 2)
    n_w = max(1, (w - segment_size) // step + 2)
    
    # Extract filename without extension
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    # Segment the image
    segment_count = 0
    for i in range(n_h):
        for j in range(n_w):
            # Calculate top-left corner of segment
            y = min(i * step, max(0, h - segment_size))
            x = min(j * step, max(0, w - segment_size))
            
            # Handle edge cases: if near image edge, adjust to get full segment size
            if y + segment_size > h:
                y = max(0, h - segment_size)
            if x + segment_size > w:
                x = max(0, w - segment_size)
            
            # Extract segment - ensure we don't go beyond image bounds
            y_end = min(y + segment_size, h)
            x_end = min(x + segment_size, w)
            segment = img[y:y_end, x:x_end]
            
            # If segment is smaller than expected (edge case), pad it
            if segment.shape[0] < segment_size or segment.shape[1] < segment_size:
                padded_segment = np.ones((segment_size, segment_size, 3), dtype=np.uint8) * 255
                padded_segment[:segment.shape[0], :segment.shape[1], :] = segment
                segment = padded_segment
            
            # Skip empty/white segments: check if enough content
            # Convert to grayscale
            segment_gray = cv2.cvtColor(segment, cv2.COLOR_BGR2GRAY)
            # Calculate the percentage of black pixels
            black_ratio = np.sum(segment_gray < 128) / (segment_size * segment_size)
            
            if black_ratio < min_content:
                continue
                
            # Save segment
            segment_path = os.path.join(output_dir, f"{base_name}_seg_{segment_count:04d}.png")
            cv2.imwrite(segment_path, segment)
            
            segment_count += 1
    
    return segment_count

def process_all_cad_drawings(input_dir, output_dir, segment_size=1024, overlap=0.3, min_content=0.02):
    """
    Process all CAD drawings in the input directory.
    
    Args:
        input_dir: Directory containing CAD drawings
        output_dir: Directory to save segments
        segment_size: Size of each segment (square)
        overlap: Percentage of overlap between segments
        min_content: Minimum content percentage to keep a segment
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all CAD drawings in input directory
    input_files = list(Path(input_dir).glob("*.png")) + \
                  list(Path(input_dir).glob("*.jpg")) + \
                  list(Path(input_dir).glob("*.jpeg")) + \
                  list(Path(input_dir).glob("*.tif")) + \
                  list(Path(input_dir).glob("*.tiff"))
    
    print(f"Found {len(input_files)} CAD drawings in {input_dir}")
    
    # Process each drawing
    total_segments = 0
    skipped_files = 0
    
    for input_file in tqdm(input_files, desc="Processing CAD drawings"):
        try:
            segments = segment_large_cad(input_file, output_dir, segment_size, overlap, min_content)
            total_segments += segments
            
            if segments == 0:
                skipped_files += 1
                print(f"Warning: No segments created for {input_file}")
        except Exception as e:
            print(f"Error processing {input_file}: {e}")
            skipped_files += 1
    
    print(f"Segmentation complete!")
    print(f"Total segments created: {total_segments}")
    print(f"Files processed: {len(input_files) - skipped_files}/{len(input_files)}")
    print(f"Segments saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Segment CAD drawings into smaller tiles')
    parser.add_argument('-i', '--input', type=str, default='Pattern_match/input_CAD',
                        help='Input directory containing CAD drawings')
    parser.add_argument('-o', '--output', type=str, default='seg_cad',
                        help='Output directory for segments')
    parser.add_argument('-s', '--size', type=int, default=1024,
                        help='Segment size in pixels (default: 1024)')
    parser.add_argument('--overlap', type=float, default=0.2,
                        help='Overlap between segments as a percentage (default: 0.3)')
    parser.add_argument('--min-content', type=float, default=0.02,
                        help='Minimum content percentage to keep a segment (default: 0.02)')
    
    args = parser.parse_args()
    
    process_all_cad_drawings(
        args.input, 
        args.output, 
        args.size, 
        args.overlap,
        args.min_content
    )

if __name__ == "__main__":
    main()