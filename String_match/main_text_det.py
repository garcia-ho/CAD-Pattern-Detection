#!/usr/bin/env python3
# filepath: /home/r27user6/CAD_matching/String_match/main_text_det.py

import os
import argparse
import time
import sys
import glob
import shutil
import multiprocessing
from datetime import datetime

# Import functions from existing files
from string_detect import (
    load_target_strings,
    highlight_strings_in_pdf,
    process_all_pdfs
)

from add_ocr_layer import add_ocr_to_pdfs

def process_pdfs(input_paths=None, targets_path="String_match/target.txt", cores=8, 
                dpi=400, sensitivity=10, skip_preprocess=False, skip_ocr=False):
    """
    Process PDF files - preprocess, OCR, and highlight target strings
    Handles both single and multiple files with a unified approach
    
    Args:
        input_paths (list): List of paths to files or directories. If None, uses default directory.
        targets_path (str): Path to file containing target strings
        cores (int): Number of CPU cores to use
        dpi (int): DPI resolution for OCR
        sensitivity (int): Detection sensitivity
        skip_preprocess (bool): Whether to skip preprocessing
        skip_ocr (bool): Whether to skip OCR
    """
    # Setup paths
    filtered_dir = "String_match/filtered_img"
    output_dir = "String_match/output"
    
    # Create necessary directories
    os.makedirs(filtered_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # If no input paths provided, use default directory
    if not input_paths:
        input_paths = ["String_match/input_pdf"]
    
    # Create log file
    log_path = os.path.join(output_dir, "processing_log.txt")
    with open(log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f"\n{'='*70}\n")
        log_file.write(f"PDF PROCESSING SESSION - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"{'='*70}\n\n")
    
    # Load target strings
    target_strings = load_target_strings(targets_path)
    
    # Log session info
    with open(log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f"Target strings: {', '.join(target_strings)}\n")
        log_file.write(f"Processing options: {cores} cores, {dpi} DPI, sensitivity {sensitivity}\n")
        if skip_preprocess:
            log_file.write("Skipping preprocessing step\n")
        if skip_ocr:
            log_file.write("Skipping OCR step\n")
        log_file.write(f"Input paths: {', '.join(input_paths)}\n\n")
    
    # Collect all PDF files to process
    all_pdf_files = []
    
    for input_path in input_paths:
        if os.path.isdir(input_path):
            # If directory, find all PDFs inside
            pdfs = glob.glob(os.path.join(input_path, "*.pdf"))
            all_pdf_files.extend(pdfs)
        elif os.path.isfile(input_path) and input_path.lower().endswith('.pdf'):
            # If file and it's a PDF, add it
            all_pdf_files.append(input_path)
        elif not os.path.exists(input_path):
            # Try with default prefix
            alt_path = os.path.join("String_match/input_pdf", os.path.basename(input_path))
            if os.path.exists(alt_path) and alt_path.lower().endswith('.pdf'):
                all_pdf_files.append(alt_path)
            else:
                with open(log_path, 'a', encoding='utf-8') as log_file:
                    log_file.write(f"Error: File not found: {input_path}\n")
                print(f"Error: File not found: {input_path}")
    
    # Log the files found
    with open(log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f"Found {len(all_pdf_files)} PDF files to process\n")
        for pdf_file in all_pdf_files:
            log_file.write(f"- {pdf_file}\n")
        log_file.write("\n")
    
    if not all_pdf_files:
        with open(log_path, 'a', encoding='utf-8') as log_file:
            log_file.write("No PDF files found to process. Exiting.\n")
        print("No PDF files found to process.")
        return
    
    # Start timing
    start_time = time.time()
    
    # STEP 1: Copy or preprocess PDFs
    with open(log_path, 'a', encoding='utf-8') as log_file:
        log_file.write("STEP 1: Preparing PDFs for OCR\n")
        log_file.write("--------------------------\n")
    
    # Simple copy for all PDFs (since they're all single-page)
    for pdf_file in all_pdf_files:
        base_filename = os.path.basename(pdf_file)
        filtered_path = os.path.join(filtered_dir, base_filename)
        try:
            shutil.copy2(pdf_file, filtered_path)
            with open(log_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f"Copied: {base_filename}\n")
        except Exception as e:
            with open(log_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f"Error copying {base_filename}: {e}\n")
    
    with open(log_path, 'a', encoding='utf-8') as log_file:
        log_file.write("PDF preparation complete.\n\n")
    
    # STEP 2: Add OCR layer
    if not skip_ocr:
        with open(log_path, 'a', encoding='utf-8') as log_file:
            log_file.write("STEP 2: Adding OCR Layer\n")
            log_file.write("---------------------\n")
        
        # Redirect stdout to log file for OCR
        original_stdout = sys.stdout
        log_capture = open(log_path, 'a', encoding='utf-8')
        sys.stdout = log_capture
        
        ocr_results = add_ocr_to_pdfs(filtered_dir)
        
        # Restore stdout
        sys.stdout = original_stdout
        log_capture.close()
        
        with open(log_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f"\nOCR layer addition complete: {ocr_results['success']} successful, {ocr_results['failed']} failed\n\n")
    else:
        with open(log_path, 'a', encoding='utf-8') as log_file:
            log_file.write("STEP 2: OCR Layer Addition (SKIPPED)\n\n")
    
    # STEP 3: Highlight target strings
    with open(log_path, 'a', encoding='utf-8') as log_file:
        log_file.write("STEP 3: Highlighting Target Strings\n")
        log_file.write("------------------------------\n")
    
    # Redirect stdout to log file for highlighting
    original_stdout = sys.stdout
    log_capture = open(log_path, 'a', encoding='utf-8')
    sys.stdout = log_capture
    
    results = process_all_pdfs(
        input_folder=filtered_dir,
        output_folder=output_dir,
        target_file=targets_path,
        max_workers=cores,
        dpi=dpi,
        sensitivity=sensitivity
    )
    
    # Restore stdout
    sys.stdout = original_stdout
    log_capture.close()
    
    # Collect statistics
    total_files = len(all_pdf_files)
    processed_files = len([r for r in results if r.get('success', False)]) if results else 0
    failed_files = total_files - processed_files
    
    total_occurrences = 0
    if results:
        for result in results:
            occurrences = result.get('occurrences', {})
            total_occurrences += sum(occurrences.values())
    
    # End timing
    end_time = time.time()
    total_time = end_time - start_time
    
    # Write summary
    with open(log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f"\n{'='*70}\n")
        log_file.write("PROCESSING SUMMARY\n")
        log_file.write(f"{'='*70}\n\n")
        log_file.write(f"Total PDF files: {total_files}\n")
        log_file.write(f"Successfully processed: {processed_files}\n")
        log_file.write(f"Failed: {failed_files}\n")
        log_file.write(f"Total target occurrences found: {total_occurrences}\n")
        log_file.write(f"Total processing time: {total_time:.2f} seconds\n\n")
    
    print(f"\nProcessing complete!")
    print(f"Files processed: {processed_files}/{total_files}")
    print(f"Total target occurrences found: {total_occurrences}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Log file saved to: {log_path}")
    
    return {
        'total_files': total_files,
        'processed_files': processed_files,
        'failed_files': failed_files,
        'total_occurrences': total_occurrences,
        'total_time': total_time
    }

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Process PDFs: Preprocess, OCR, and highlight target strings")
    parser.add_argument("pdf_file", nargs="*", help="Optional: specific PDF file(s) to process")
    parser.add_argument("--targets", default="String_match/target.txt", 
                      help="File containing target strings (one per line)")
    parser.add_argument("--cores", type=int, default=8, 
                      help="Number of CPU cores to use (default: 8)")
    parser.add_argument("--dpi", type=int, default=400, 
                      help="DPI resolution for OCR (higher = more accurate but slower, default: 400)")
    parser.add_argument("--sensitivity", type=int, default=10,
                      help="Detection sensitivity (1-10, where 1 is strict and 10 is permissive, default: 10)")
    parser.add_argument("--skip-preprocess", action="store_true", 
                      help="Skip preprocessing step")
    parser.add_argument("--skip-ocr", action="store_true", 
                      help="Skip OCR processing (use this if PDFs already have OCR layer)")

    args = parser.parse_args()
    
    print("PDF Processing Pipeline: Preprocess, OCR, and Target String Detection")
    print("==================================================================")
    print(f"Using {args.cores} CPU cores, {args.dpi} DPI, sensitivity level {args.sensitivity}")
    
    # Process files (either specific files or all files)
    process_pdfs(
        input_paths=args.pdf_file if args.pdf_file else None,
        targets_path=args.targets,
        cores=args.cores,
        dpi=args.dpi,
        sensitivity=args.sensitivity,
        skip_preprocess=args.skip_preprocess,
        skip_ocr=args.skip_ocr
    )

if __name__ == "__main__":
    main()