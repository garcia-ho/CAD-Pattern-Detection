import cv2
import os
import numpy as np
import concurrent.futures
from functools import partial
from detector import PatternDetector

# Define a parallel foreach function with dopar-like behavior
def foreach_dopar(iterable, func, max_workers=4):
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

def main(detect_rotations=True):
    """
    Main function to process images and detect patterns
    
    Args:
        detect_rotations: Whether to detect rotated patterns (defaults to True)
    """
    # Determine rotations to use
    rotations = [0, 90, 180, 270] if detect_rotations else [0]
    print(f"Detecting patterns with rotations: {rotations}")

    # Initialize the pattern detector with multi-scale and multi-rotation support
    # Remove the overlap_threshold parameter that's causing the error
    detector = PatternDetector(
        threshold=0.39,        # Lower threshold to allow for rotation and scaling variations
        scale_range=(0.01, 0.2), # Search from 10% to 60% of original size
        scale_steps = 10,        # Use 5 scale steps for efficiency
        rotations=rotations    # Specify rotations to check
    )
    
    # Load the template pattern
    template_path = "input_target/targetA.png"
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
    test_images = ["testA.png", "testB.png"]
    
    # Create partial function with fixed parameters
    process_func = partial(
        process_image, 
        template=template, 
        detector=detector, 
        input_dir=input_dir, 
        output_dir=output_dir
    )
    
    # Process images in parallel using foreach_dopar (similar to R's foreach %dopar%)
    results = foreach_dopar(test_images, process_func, max_workers=16)
    
    # Print summary of results
    print("\nSummary of results:")
    for image_name, count in results:
        print(f"  {image_name}: {count} matches")
    
    print("All images processed successfully")
    
    return results

if __name__ == "__main__":
    # Set to False to skip rotation detection and save processing time
    detect_rotations = False
    main(detect_rotations)