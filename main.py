import os
import argparse
import logging
import sys
from pathlib import Path
from src.detector import CADObjectDetector

# Create logs directory if it doesn't exist
logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(logs_dir, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(os.path.join(logs_dir, "cad_recognition.log")),
                        logging.StreamHandler(sys.stdout)
                    ])
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='CAD Pattern Matching for Line-based Technical Diagrams')
    
    # Use default folders in project structure
    default_cad = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'input_CAD')
    default_target = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'input_target')
    default_output = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output_CAD')
    
    parser.add_argument('--cad', default=default_cad, 
                        help=f'Path to folder containing CAD PNG files (default: input_CAD)')
    parser.add_argument('--target', default=default_target, 
                        help=f'Path to folder containing target PNG images (default: input_target)')
    parser.add_argument('--output', default=default_output, 
                        help=f'Path to save output files (default: output_CAD)')
    parser.add_argument('--method', default='hybrid', 
                        choices=['pattern', 'geometric', 'hybrid'],
                        help='Matching method to use (pattern=template matching, geometric=feature-based, hybrid=both)')
    parser.add_argument('--threshold', type=float, default=0.75, 
                        help='Matching threshold (0.0-1.0)')
    parser.add_argument('--sensitivity', type=str, default='medium',
                        choices=['low', 'medium', 'high', 'very-high'],
                        help='Detection sensitivity preset (default: medium)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
                        
    return parser.parse_args()

def ensure_directories_exist():
    """Create default directories if they don't exist"""
    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define default directories
    default_dirs = [
        os.path.join(project_dir, 'input_CAD'),
        os.path.join(project_dir, 'input_target'),
        os.path.join(project_dir, 'output_CAD'),
        os.path.join(project_dir, 'logs')
    ]
    
    # Create directories if they don't exist
    for directory in default_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

def print_results(results):
    """Print detection results to the console"""
    print("\n===== CAD RECOGNITION RESULTS =====\n")
    
    for target_name, target_data in results.items():
        print(f"Target: {target_name}")
        print("-" * (len(f"Target: {target_name}")))
        
        for cad_name, cad_data in target_data.items():
            print(f"  CAD file: {cad_name}")
            print(f"    Occurrences found: {cad_data['count']}")
            if 'error' in cad_data and cad_data['error']:
                print(f"    Error: {cad_data['error']}")
            if cad_data['highlighted_path']:
                print(f"    Highlighted image: {cad_data['highlighted_path']}")
            else:
                print("    No highlighted image generated.")
            print()
        print()

def main():
    """Main entry point"""
    # Create default directories if needed
    ensure_directories_exist()
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Apply sensitivity presets to threshold
    if args.sensitivity == 'low':
        args.threshold = 0.8
    elif args.sensitivity == 'medium':
        args.threshold = 0.7
    elif args.sensitivity == 'high':
        args.threshold = 0.6
    elif args.sensitivity == 'very-high':
        args.threshold = 0.5
    
    logger.info("CAD Recognition Tool starting...")
    logger.info(f"CAD folder: {args.cad}")
    logger.info(f"Target folder: {args.target}")
    logger.info(f"Output folder: {args.output}")
    logger.info(f"Method: {args.method}")
    logger.info(f"Threshold: {args.threshold}")
    logger.info(f"Sensitivity: {args.sensitivity}")
    
    # Check if the necessary input directories have files
    cad_files = [f for f in os.listdir(args.cad) if f.lower().endswith('.png')]
    target_files = [f for f in os.listdir(args.target) if f.lower().endswith('.png')]
    
    if not cad_files:
        logger.error(f"No PNG files found in {args.cad}")
        logger.error("Please add CAD PNG files to the input_CAD directory.")
        return {}
    
    if not target_files:
        logger.error(f"No PNG target files found in {args.target}")
        logger.error("Please add target PNG files to the input_target directory.")
        return {}
    
    # Process all CAD files and targets
    detector = CADObjectDetector()
    results = {}
    for cad_file in cad_files:
        for target_file in target_files:
            cad_path = os.path.join(args.cad, cad_file)
            target_path = os.path.join(args.target, target_file)
            output_path = os.path.join(args.output, f"{os.path.splitext(cad_file)[0]}_matched_{os.path.splitext(target_file)[0]}.png")
            try:
                match_count, highlighted_path = detector.detect_objects(
                    cad_path, 
                    target_path, 
                    output_path, 
                    method=args.method,
                    threshold=args.threshold
                )
                if cad_file not in results:
                    results[cad_file] = {}
                results[cad_file][target_file] = {
                    'count': match_count,
                    'highlighted_path': highlighted_path,
                    'error': None
                }
            except Exception as e:
                logger.error(f"Error processing {cad_file} with {target_file}: {e}")
                if cad_file not in results:
                    results[cad_file] = {}
                results[cad_file][target_file] = {
                    'count': 0,
                    'highlighted_path': None,
                    'error': str(e)
                }
    
    # Print results to console
    print_results(results)
    
    logger.info("Processing complete!")
    
    return results

if __name__ == "__main__":
    main()