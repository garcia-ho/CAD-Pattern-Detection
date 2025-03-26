#!/usr/bin/env python3
# filepath: /home/r27user6/CAD_matching/String_match/main_text_det.py

import os
import argparse
import time
import sys
from datetime import datetime

# Import preprocessing functions
sys.path.append('/home/r27user6/CAD_matching')
from String_match.preprocess_text import (
    convert_pdf_to_image, 
    advanced_preprocess,
    process_pdfs
)

# Import detection functions
from String_match.string_detect import (
    load_target_words,
    process_all_images
)


def check_directories(input_dir, filtered_dir, output_dir, create=True):
    """
    Check if required directories exist and create them if needed
    
    Args:
        input_dir (str): Input directory for PDFs
        filtered_dir (str): Directory for filtered images
        output_dir (str): Output directory for results
        create (bool): Whether to create directories if they don't exist
        
    Returns:
        bool: True if all directories exist or were created
    """
    directories = [input_dir, filtered_dir, output_dir]
    
    for directory in directories:
        if not os.path.exists(directory):
            if create:
                print(f"Creating directory: {directory}")
                try:
                    os.makedirs(directory, exist_ok=True)
                except Exception as e:
                    print(f"Error creating directory {directory}: {e}")
                    return False
            else:
                print(f"Directory does not exist: {directory}")
                return False
    
    return True


def run_full_pipeline(input_dir="String_match/input_pdf", 
                    filtered_dir="String_match/filtered_img",
                    output_dir="String_match/output", 
                    target_file="String_match/target.txt",
                    dpi=400, patch_size=1024, overlap_percent=15,
                    case_sensitive=False, preprocess_only=False,
                    ocr_only=False):
    """
    Run the full pipeline - preprocess PDFs and detect text
    
    Args:
        input_dir (str): Input directory for PDFs
        filtered_dir (str): Directory for filtered images
        output_dir (str): Output directory for results
        target_file (str): Path to file containing target words
        dpi (int): DPI for PDF to image conversion
        patch_size (int): Size of image patches
        overlap_percent (int): Percentage of overlap between patches
        case_sensitive (bool): Whether to use case-sensitive matching
        preprocess_only (bool): Only run preprocessing
        ocr_only (bool): Only run OCR detection
    """
    start_time = time.time()
    
    # Check if directories exist
    if not check_directories(input_dir, filtered_dir, output_dir):
        return
    
    # Step 1: Preprocess PDFs to images
    if not ocr_only:
        print("\n" + "="*60)
        print("STEP 1: PREPROCESSING PDFs")
        print("="*60)
        
        preprocess_start = time.time()
        
        # Run preprocessing
        process_pdfs(
            input_dir=input_dir,
            output_dir=filtered_dir,
            dpi=dpi,
            advanced=True
        )
        
        preprocess_time = time.time() - preprocess_start
        print(f"Preprocessing completed in {preprocess_time:.2f} seconds")
    
    # Exit if preprocess only
    if preprocess_only:
        total_time = time.time() - start_time
        print(f"\nPreprocessing only mode. Total time: {total_time:.2f} seconds")
        return
    
    # Step 2: OCR Detection in processed images
    if not preprocess_only:
        print("\n" + "="*60)
        print("STEP 2: OCR DETECTION")
        print("="*60)
        
        ocr_start = time.time()
        
        # Check if target file exists
        if not os.path.exists(target_file):
            print(f"Error: Target file not found: {target_file}")
            return
        
        # Run OCR detection
        process_all_images(
            input_dir=filtered_dir,
            output_dir=output_dir,
            target_file=target_file,
            patch_size=patch_size,
            overlap_percent=overlap_percent,
            case_sensitive=case_sensitive
        )
        
        ocr_time = time.time() - ocr_start
        print(f"OCR detection completed in {ocr_time:.2f} seconds")
    
    # Print summary
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Input directory: {input_dir}")
    print(f"Filtered images directory: {filtered_dir}")
    print(f"Output directory: {output_dir}")
    if not ocr_only:
        print(f"PDF to image conversion DPI: {dpi}")
    if not preprocess_only:
        print(f"Target file: {target_file}")
        print(f"Patch size: {patch_size}")
        print(f"Patch overlap: {overlap_percent}%")
        print(f"Case-sensitive matching: {case_sensitive}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print("="*60)


def main():
    """Parse command line arguments and run the pipeline"""
    parser = argparse.ArgumentParser(description="Process PDFs and detect text")
    
    # Directory options
    parser.add_argument("--input-dir", default="String_match/input_pdf",
                      help="Directory containing input PDFs")
    parser.add_argument("--filtered-dir", default="String_match/filtered_img",
                      help="Directory for filtered images")
    parser.add_argument("--output-dir", default="String_match/output",
                      help="Directory for output results")
    
    # Processing options
    parser.add_argument("--target-file", default="String_match/target.txt",
                      help="Path to file containing target words")
    parser.add_argument("--dpi", type=int, default=400,
                      help="DPI for PDF to image conversion (default: 400)")
    parser.add_argument("--patch-size", type=int, default=1024,
                      help="Size of image patches (default: 1024)")
    parser.add_argument("--overlap", type=int, default=15,
                      help="Percentage of overlap between patches (default: 15)")
    parser.add_argument("--case-sensitive", action="store_true",
                      help="Use case-sensitive matching")
    
    # Mode options
    parser.add_argument("--preprocess-only", action="store_true",
                      help="Only run preprocessing step")
    parser.add_argument("--ocr-only", action="store_true",
                      help="Only run OCR detection step")
    
    args = parser.parse_args()
    
    # Run the pipeline
    run_full_pipeline(
        input_dir=args.input_dir,
        filtered_dir=args.filtered_dir,
        output_dir=args.output_dir,
        target_file=args.target_file,
        dpi=args.dpi,
        patch_size=args.patch_size,
        overlap_percent=args.overlap,
        case_sensitive=args.case_sensitive,
        preprocess_only=args.preprocess_only,
        ocr_only=args.ocr_only
    )


if __name__ == "__main__":
    main()