import os
import argparse
from trained_YOLO import load_model, YOLO_process_directory 
from trained_YOLO import generate_csv_report
from preprocessing import preprocess_directory 


HOME = '/home/r27user6/CAD_matching/Pattern_match'


def main():
    """
    Main function for the CAD image processing pipeline:
    1. Convert PDFs to PNGs and preprocess them
    2. Detect objects using YOLO
    3. Generate a report
    """
    parser = argparse.ArgumentParser(description="CAD Drawing Processing Pipeline")
    
    # Input and output options
    parser.add_argument('--input-cad', type=str, default=f'{HOME}/input_CAD',
                        help='Input directory containing CAD drawings (PDF/images)')
    parser.add_argument('--processed-dir', type=str, default=f'{HOME}/input_image',
                        help='Directory for preprocessed images')
    parser.add_argument('--output', type=str, default=f'{HOME}/output_CAD',
                        help='Output directory for annotated images')
    parser.add_argument('--model', type=str, default=f'{HOME}/runs/detect/train/weights/best.pt',
                        help='Path to YOLO model')
    
    # Preprocessing options
    parser.add_argument('--dpi', type=int, default=400,
                        help='Resolution for PDF conversion (dots per inch)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of worker threads for parallel processing')
    
    # YOLO detection options
    parser.add_argument('--size', type=int, default=1024,
                        help='Segment size (default: 1024)')
    parser.add_argument('--overlap', type=float, default=0.15,
                        help='Overlap between segments (default: 0.15)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold (default: 0.25)')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold (default: 0.45)')
    
    # Report options
    parser.add_argument('--report', type=str, default=f'{HOME}/detection_report.csv',
                        help='Path for CSV report')
    parser.add_argument('--skip-preprocess', action='store_true',
                        help='Skip preprocessing step (use existing processed images)')
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs(args.processed_dir, exist_ok=True)
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    
    # Step 1: Preprocess CAD drawings
    if not args.skip_preprocess:
        print("\n=== Step 1: Preprocessing CAD Drawings ===")
        preprocessed_images = preprocess_directory(
            args.input_cad,
            args.processed_dir,
            dpi=args.dpi,
            workers=args.workers
        )
        print(f"Preprocessed {len(preprocessed_images)} images")
    else:
        print("\n=== Skipping preprocessing (using existing processed images) ===")
    
    # Step 2: Run YOLO detection
    print("\n=== Step 2: Running YOLO Detection ===")
    all_counts, all_detections = YOLO_process_directory(
        args.processed_dir,
        args.output,
        args.model,
        args.size,
        args.overlap,
        args.conf,
        args.iou
    )
    
    # Step 3: Generate CSV report
    print("\n=== Step 3: Generating Report ===")
    if all_counts:
        generate_csv_report(all_counts, args.report)
        print(f"Detection results saved to {args.output}")
        print(f"Detection report saved to {args.report}")
    else:
        print("No detections to report")
    
    print("\n=== Processing Complete ===")
    print(f"CAD drawings from: {args.input_cad}")
    print(f"Preprocessed images: {args.processed_dir}")
    print(f"Detected results: {args.output}")
    print(f"Detection report: {args.report}")

if __name__ == "__main__":
    main()