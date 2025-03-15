import cv2
import os
import numpy as np
import concurrent.futures
import argparse
from functools import partial
from detector import PatternDetector
from datetime import datetime

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
    parser.add_argument('--min-scale', type=float, default=0.01,
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
    template_path = "input_target/butterfly_valve.png"
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
    test_images = ["test_butterfly.png"]
    
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