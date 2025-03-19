#!/usr/bin/env python3
# filepath: /home/r27user6/CAD_matching/String_match/add_ocr_layer.py

import os
import subprocess
import glob
import time
from pathlib import Path
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

def ocr_single_pdf(pdf_file):
    """Process a single PDF file with OCRmyPDF"""
    filename = os.path.basename(pdf_file)
    temp_output = pdf_file + ".ocr.pdf"
    
    try:
        # Run OCRmyPDF with force-ocr to ensure text layer is added
        cmd = [
            "ocrmypdf",
            "--force-ocr",       # Force OCR even if text exists
            "--optimize", "3",   # Level 3 optimization for smaller file size
            "--output-type", "pdf",  # Ensure PDF output
            "--quiet",           # Less verbose output
            pdf_file,            # Input file
            temp_output          # Temporary output
        ]
        
        # Execute OCRmyPDF
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Replace original with OCR'd version
            os.replace(temp_output, pdf_file)
            return {
                'filename': filename,
                'success': True,
                'stderr': None
            }
        else:
            # Clean up temporary file if it exists
            if os.path.exists(temp_output):
                os.remove(temp_output)
            return {
                'filename': filename,
                'success': False,
                'stderr': result.stderr
            }
    except Exception as e:
        # Clean up temporary file if it exists
        if os.path.exists(temp_output):
            os.remove(temp_output)
        return {
            'filename': filename,
            'success': False,
            'stderr': str(e)
        }

def add_ocr_to_pdfs(input_dir="String_match/filtered_img"):
    """
    Add OCR text layer to all PDFs in the input directory, overwriting the original files.
    Uses parallel processing for faster execution.
    
    Args:
        input_dir (str): Directory containing PDFs to process
    
    Returns:
        dict: Summary of processing results
    """
    # Ensure the directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Directory '{input_dir}' not found.")
        return {'success': 0, 'failed': 0, 'total': 0}
    
    # Find all PDF files
    pdf_files = glob.glob(os.path.join(input_dir, "*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return {'success': 0, 'failed': 0, 'total': 0}
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Get CPU count for parallel processing
    cpu_count = multiprocessing.cpu_count()
    max_workers = min(cpu_count, len(pdf_files))  # Use at most one job per file
    
    print(f"Using {max_workers} parallel processes")
    
    # Process each PDF in parallel
    success_count = 0
    failure_count = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(ocr_single_pdf, pdf_file): pdf_file for pdf_file in pdf_files}
        
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            filename = result['filename']
            
            print(f"[{i+1}/{len(pdf_files)}] Processing {filename}...")
            
            if result['success']:
                success_count += 1
                print(f"  ✓ Successfully added OCR text layer")
            else:
                failure_count += 1
                print(f"  ✗ Failed to add OCR text layer")
                if result['stderr']:
                    print(f"  Error: {result['stderr']}")
    
    print("\nOCR Processing complete!")
    print(f"Successfully processed: {success_count}/{len(pdf_files)}")
    print(f"Failed: {failure_count}/{len(pdf_files)}")
    
    return {
        'success': success_count,
        'failed': failure_count,
        'total': len(pdf_files)
    }

if __name__ == "__main__":
    print("This module contains functions for adding OCR layers to PDFs.")
    print("Please use main_text_det.py to run the full processing pipeline.")