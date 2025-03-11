import os
import sys
import argparse

# Add the current directory to the path to ensure detector.py can be imported
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

# Now import the detector module
from detector import detect_objects_in_cad

def main():
    """
    Main function to parse command line arguments and run the detector
    """
    parser = argparse.ArgumentParser(description='CAD Object Recognition')
    parser.add_argument('--cad_folder', type=str, default='input_CAD',
                      help='Folder containing CAD PDF files')
    parser.add_argument('--target_folder', type=str, default='input_target',
                      help='Folder containing target PNG images')
    parser.add_argument('--output_folder', type=str, default='output_CAD',
                      help='Folder to save highlighted CAD files')
    parser.add_argument('--method', type=str, default='feature',
                      choices=['template', 'feature', 'contour'],
                      help='Detection method to use')
    parser.add_argument('--threshold', type=float, default=0.7,
                      help='Matching threshold')
    parser.add_argument('--dpi', type=int, default=300,
                      help='DPI for rendering CAD PDFs (higher values capture more details)')
    
    args = parser.parse_args()
    
    # Create folders if they don't exist
    os.makedirs(args.cad_folder, exist_ok=True)
    os.makedirs(args.target_folder, exist_ok=True)
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Run detection
    print(f"Starting CAD object detection")
    print(f"- CAD folder: {args.cad_folder}")
    print(f"- Target folder: {args.target_folder}")
    print(f"- Output folder: {args.output_folder}")
    print(f"- Method: {args.method}")
    print(f"- Threshold: {args.threshold}")
    print(f"- DPI: {args.dpi}")
    
    try:
        results = detect_objects_in_cad(
            args.cad_folder, args.target_folder, args.output_folder, 
            args.method, args.threshold, args.dpi
        )
        
        # Print summary of results
        print("\nDetection Summary:")
        for target_name, cad_results in results.items():
            print(f"\nTarget: {target_name}")
            for cad_name, result in cad_results.items():
                print(f"  - CAD: {cad_name}, Count: {result['count']}")
                if result['highlighted_path']:
                    print(f"      Highlighted PDF: {os.path.basename(result['highlighted_path'])}")
    except Exception as e:
        print(f"Error during detection: {e}")
        # Print more detailed error information
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()