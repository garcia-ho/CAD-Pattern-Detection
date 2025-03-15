import os
import argparse
import re
from pathlib import Path
import fitz  # PyMuPDF for PDF handling
import cv2
import pytesseract
import numpy as np
from PIL import Image
import io
import concurrent.futures
import multiprocessing

def extract_text_from_pdf_with_ocr(pdf_path, output_txt_path=None):
    """
    Extract text from a PDF using OCR by converting pages to images.
    
    Args:
        pdf_path (str): Path to the PDF file.
        output_txt_path (str, optional): Path to save the extracted text.
                                       If None, will use PDF name with .txt extension.
    
    Returns:
        tuple: (Path to the output text file, extracted text)
    """
    # Default output path if not specified
    if output_txt_path is None:
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_txt_path = os.path.join("String_match/output", f"{base_name}.txt")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
    
    try:
        # Open the PDF with PyMuPDF
        doc = fitz.open(pdf_path)
        all_text = ""
        
        # Process each page
        for page_num, page in enumerate(doc):
            # Convert the page to a high-resolution image (RGB mode)
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72), alpha=False)
            
            # Convert PyMuPDF pixmap to OpenCV format
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            
            # If we have 4 channels (RGBA), convert to RGB
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            
            # OCR processing
            # Convert OpenCV image to PIL Image
            pil_img = Image.fromarray(img)
            
            # Use pytesseract to extract text from the image
            page_text = pytesseract.image_to_string(pil_img, lang='eng')
            
            # Add page text to overall text
            all_text += page_text + "\n\n"
            print(f"Processed page {page_num + 1}/{len(doc)} of {os.path.basename(pdf_path)}")
        
        # Write extracted text to file
        with open(output_txt_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(all_text)
        
        doc.close()
        print(f"Text extracted successfully from '{pdf_path}'")
        print(f"Text saved to '{output_txt_path}'")
        
        return output_txt_path, all_text
    
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None, None

def count_occurrences(text, target_string="AHU"):
    """
    Count occurrences of a target string in text.
    
    Args:
        text (str): The text to search in
        target_string (str): The string to search for
        
    Returns:
        int: Number of occurrences
    """
    if not text:
        return 0
    # Case-insensitive search for whole word
    return len(re.findall(r'\b' + re.escape(target_string) + r'\b', text, re.IGNORECASE))

def process_pdf_page(args):
    """
    Process a single PDF page to find "AHU" occurrences.
    Designed to be used with multiprocessing.
    
    Args:
        args (tuple): (doc_path, page_num, target_string, dpi)
    
    Returns:
        dict: Contains page number, found boxes, and processing information
    """
    doc_path, page_num, target_string, dpi = args
    
    try:
        # Open the document (each process needs its own handle)
        doc = fitz.open(doc_path)
        page = doc[page_num]
        
        print(f"Processing page {page_num + 1}/{len(doc)} of {os.path.basename(doc_path)}")
        
        # Convert page to high-resolution image for OCR
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72), alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        
        # Convert to RGB if needed
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        # ENHANCED IMAGE PREPROCESSING PIPELINE
        
        # 1. Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # 2. Store original grayscale for multi-processing approach
        gray_orig = gray.copy()
        
        # 3. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_clahe = clahe.apply(gray)
        
        # 4. Apply adaptive thresholding with different parameters
        # Threshold 1: Standard
        binary1 = cv2.adaptiveThreshold(gray_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 15, 9)
        
        # Threshold 2: More aggressive for small text
        binary2 = cv2.adaptiveThreshold(gray_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 9, 3)
        
        # 5. Process images to remove lines and curves near text
        # Use morphological operations to identify and remove lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        
        # Detect horizontal and vertical lines
        horizontal_lines = cv2.morphologyEx(255-binary1, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        vertical_lines = cv2.morphologyEx(255-binary1, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        
        # Combine lines
        lines = cv2.add(horizontal_lines, vertical_lines)
        
        # Remove lines from the image
        no_lines = cv2.subtract(255-binary1, lines)
        
        # 6. Create multiple processed images for better OCR accuracy
        # Image 1: Standard processing
        kernel1 = np.ones((2, 2), np.uint8)
        proc_img1 = cv2.morphologyEx(binary1, cv2.MORPH_CLOSE, kernel1)
        
        # Image 2: More aggressive processing for small text
        kernel2 = np.ones((1, 1), np.uint8)
        proc_img2 = cv2.morphologyEx(binary2, cv2.MORPH_OPEN, kernel2)
        
        # Image 3: No lines version
        proc_img3 = 255 - no_lines
        
        # 7. Convert all processed images to PIL format
        pil_img1 = Image.fromarray(proc_img1)
        pil_img2 = Image.fromarray(proc_img2)
        pil_img3 = Image.fromarray(proc_img3)
        pil_img_orig = Image.fromarray(gray_orig)
        
        # Configure tesseract for optimal text detection
        # Different PSM (Page Segmentation Modes) for different detection strategies
        configs = [
            '--oem 3 --psm 11 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -"',  # Sparse text
            '--oem 3 --psm 6 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -"',   # Uniform block of text
            '--oem 3 --psm 3 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -"',   # Full page
            '--oem 3 --psm 12 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -"'   # Single text line
        ]
        
        # Process with multiple configurations and images
        all_boxes = []
        for img_idx, pil_img in enumerate([pil_img1, pil_img2, pil_img3, pil_img_orig]):
            for config_idx, config in enumerate(configs):
                # Skip some combinations to reduce processing time
                if img_idx == 3 and config_idx > 1:  # Only use first two configs for original image
                    continue
                
                try:
                    # Get OCR data with bounding boxes
                    ocr_data = pytesseract.image_to_data(pil_img, config=config, output_type=pytesseract.Output.DICT)
                    
                    # Process OCR results
                    for i, text in enumerate(ocr_data['text']):
                        # Skip empty text
                        if not text.strip():
                            continue
                        
                        # Skip very low confidence results
                        if ocr_data['conf'][i] < -1:  # Very permissive
                            continue
                        
                        # Normalize text for matching
                        normalized_text = text.strip().upper()
                        
                        # Various matching strategies with different thresholds
                        found = False
                        match_type = ""
                        
                        # 1. Exact match
                        if target_string.upper() == normalized_text:
                            found = True
                            match_type = "exact"
                        
                        # 2. Part of word (with reasonable confidence threshold)
                        elif target_string.upper() in normalized_text and ocr_data['conf'][i] > 20:
                            found = True
                            match_type = "contains"
                        
                        # 3. Fuzzy matching for common OCR mistakes with "AHU"
                        # This handles cases like "AHL", "ARU", "AMU", etc.
                        elif len(normalized_text) == 3 and normalized_text[0] == "A" and ocr_data['conf'][i] > 10:
                            # Match H-like characters (H, M, N, K)
                            h_like = normalized_text[1] in "HMNKFPRB"
                            # Match U-like characters (U, O, 0, Q, C, G)
                            u_like = normalized_text[2] in "UO0QCGL"
                            
                            if h_like and u_like:
                                found = True
                                match_type = "fuzzy-3char"
                        
                        # 4. Check for AHU within longer text (e.g., "MAIN-AHU-01")
                        elif len(normalized_text) > 3:
                            # Look for A?U pattern where ? can be any character
                            matches = re.findall(r'A[A-Z0-9]U', normalized_text)
                            if matches:
                                found = True
                                match_type = "pattern"
                        
                        if found:
                            # Get the bounding box coordinates from OCR data
                            x = ocr_data['left'][i]
                            y = ocr_data['top'][i]
                            w = ocr_data['width'][i]
                            h = ocr_data['height'][i]
                            conf = ocr_data['conf'][i]
                            
                            # Add box info with source info for deduplication
                            all_boxes.append({
                                'x': x, 'y': y, 'w': w, 'h': h, 
                                'text': text, 'conf': conf,
                                'match_type': match_type,
                                'source': f"img{img_idx}-config{config_idx}"
                            })
                except Exception as e:
                    print(f"OCR error with configuration {config_idx} on image {img_idx} on page {page_num+1}: {e}")
        
        # Deduplicate boxes by merging overlapping ones
        merged_boxes = []
        if all_boxes:
            # Sort by confidence
            all_boxes = sorted(all_boxes, key=lambda box: box['conf'], reverse=True)
            
            for box in all_boxes:
                # Check if this box overlaps with any already merged box
                is_new = True
                box_rect = (box['x'], box['y'], box['x'] + box['w'], box['y'] + box['h'])
                
                for idx, merged_box in enumerate(merged_boxes):
                    merged_rect = (merged_box['x'], merged_box['y'], 
                                   merged_box['x'] + merged_box['w'], merged_box['y'] + merged_box['h'])
                    
                    # Calculate intersection
                    x1 = max(box_rect[0], merged_rect[0])
                    y1 = max(box_rect[1], merged_rect[1])
                    x2 = min(box_rect[2], merged_rect[2])
                    y2 = min(box_rect[3], merged_rect[3])
                    
                    # Check for overlap
                    if x1 < x2 and y1 < y2:
                        overlap_area = (x2 - x1) * (y2 - y1)
                        box_area = box['w'] * box['h']
                        merged_area = merged_box['w'] * merged_box['h']
                        
                        # If significant overlap, merge boxes by taking the union
                        if overlap_area > 0.5 * min(box_area, merged_area):
                            is_new = False
                            # Keep the higher confidence data but expand the box to contain both
                            if box['conf'] > merged_box['conf']:
                                merged_box['text'] = box['text']
                                merged_box['conf'] = box['conf']
                                merged_box['match_type'] = box['match_type']
                            
                            # Expand to union
                            merged_box['x'] = min(box_rect[0], merged_rect[0])
                            merged_box['y'] = min(box_rect[1], merged_rect[1])
                            merged_box['w'] = max(box_rect[2], merged_rect[2]) - merged_box['x']
                            merged_box['h'] = max(box_rect[3], merged_rect[3]) - merged_box['y']
                            break
                
                # Add new box if not merged
                if is_new:
                    merged_boxes.append(box)
        
        # Close the document
        doc.close()
        
        return {
            'page_num': page_num,
            'boxes': merged_boxes,
            'scale': 72 / dpi  # Need this for coordinate conversion
        }
        
    except Exception as e:
        print(f"Error processing page {page_num+1}: {e}")
        return {
            'page_num': page_num,
            'boxes': [],
            'error': str(e)
        }

def highlight_ahu_in_pdf_parallel(input_pdf, output_pdf, target_string="AHU", max_workers=None):
    """
    Find and highlight occurrences of "AHU" in PDF with red transparent boxes.
    Uses parallel processing for improved performance.
    
    Args:
        input_pdf (str): Path to input PDF
        output_pdf (str): Path to save highlighted PDF
        target_string (str): String to highlight
        max_workers (int): Maximum number of parallel workers (None = auto)
    
    Returns:
        int: Total number of occurrences found
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    
    # Get CPU count if max_workers is None
    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free
    
    # Open the PDF to get page count
    doc = fitz.open(input_pdf)
    num_pages = len(doc)
    doc.close()
    
    print(f"Processing {os.path.basename(input_pdf)} with {max_workers} workers")
    print(f"Total pages: {num_pages}")
    
    # Prepare tasks for parallel processing
    tasks = [(input_pdf, i, target_string, 600) for i in range(num_pages)]
    
    # Process pages in parallel
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_pdf_page, tasks))
    
    # Open the PDF again for highlighting (must be done in main process)
    doc = fitz.open(input_pdf)
    
    # Track total occurrences
    total_occurrences = 0
    
    # Apply highlights based on results
    for result in results:
        if not result or 'error' in result:
            continue
            
        page_num = result['page_num']
        boxes = result['boxes']
        scale = result['scale']
        
        page = doc[page_num]
        
        # Add highlights to PDF
        for box in boxes:
            # Add padding around the box for better visibility
            padding = int(box['h'] * 0.3)  # 30% padding
            x = max(0, box['x'] - padding)
            y = max(0, box['y'] - padding)
            w = box['w'] + padding * 2
            h = box['h'] + padding * 2
            
            # Convert OCR coordinates to PDF coordinates
            pdf_x = x * scale
            pdf_y = y * scale
            pdf_w = w * scale
            pdf_h = h * scale
            
            # Create rectangle annotation with fill
            rect = fitz.Rect(pdf_x, pdf_y, pdf_x + pdf_w, pdf_y + pdf_h)
            highlight = page.add_rect_annot(rect)
            highlight.set_colors(stroke=(1, 0, 0), fill=(1, 0.8, 0.8))  # Red outline, light red fill
            highlight.set_opacity(0.4)  # Transparency
            highlight.update()
            
            total_occurrences += 1
            print(f"  - Found '{box['text']}' on page {page_num + 1} (confidence: {box['conf']:.1f}, type: {box['match_type']})")
    
    # Save the highlighted PDF
    doc.save(output_pdf)
    doc.close()
    
    print(f"Total occurrences found: {total_occurrences}")
    print(f"Highlighted PDF saved to '{output_pdf}'")
    
    return total_occurrences

def process_all_pdfs(input_folder="String_match/input_pdf", output_folder="String_match/output", max_workers=None):
    """
    Process all PDFs in the input folder, find AHU occurrences and highlight them.
    Uses parallel processing for better performance.
    
    Args:
        input_folder (str): Path to folder containing PDFs
        output_folder (str): Path to save highlighted PDFs and report
        max_workers (int): Maximum number of workers per PDF
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Get optimal number of workers if not specified
    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free
    
    # Get list of PDFs to process
    pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {input_folder}")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process")
    print(f"Using up to {max_workers} CPU cores per PDF")
    
    # Create a report file
    report_path = os.path.join(output_folder, "ahu_detection_report.txt")
    with open(report_path, 'w', encoding='utf-8') as report:
        report.write("AHU Detection Report\n")
        report.write("===================\n\n")
        
        total_occurrences = 0
        processed_files = 0
        
        # Process each PDF in the input folder
        for filename in pdf_files:
            input_path = os.path.join(input_folder, filename)
            base_name = os.path.splitext(filename)[0]
            highlighted_pdf_path = os.path.join(output_folder, f"{base_name}_highlighted.pdf")
            
            print(f"\nProcessing {filename}...")
            
            # Find and highlight AHU occurrences
            occurrences = highlight_ahu_in_pdf_parallel(input_path, highlighted_pdf_path, max_workers=max_workers)
            
            processed_files += 1
            total_occurrences += occurrences
            
            # Write to report
            report.write(f"File: {filename}\n")
            report.write(f"AHU occurrences: {occurrences}\n\n")
        
        # Write summary
        report.write("\nSummary\n")
        report.write("-------\n")
        report.write(f"Total files processed: {processed_files}\n")
        report.write(f"Total AHU occurrences across all files: {total_occurrences}\n")
    
    print(f"\nReport generated at: {report_path}")

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Find and highlight 'AHU' occurrences in PDFs using OCR")
    parser.add_argument("pdf_file", nargs="?", help="Optional: specific PDF file to process")
    parser.add_argument("--cores", type=int, default=16, 
                        help="Number of CPU cores to use (default: auto-detect)")
    
    args = parser.parse_args()
    
    if args.pdf_file:
        # Process single file
        input_path = os.path.join("String_match/input_pdf", args.pdf_file)
        if not os.path.exists(input_path):
            print(f"Error: File not found: {input_path}")
        else:
            base_name = os.path.splitext(args.pdf_file)[0]
            output_pdf = os.path.join("String_match/output", f"{base_name}_highlighted.pdf")
            
            os.makedirs("String_match/output", exist_ok=True)
            highlight_ahu_in_pdf_parallel(input_path, output_pdf, max_workers=args.cores)
    else:
        # Process all PDFs in the folder
        process_all_pdfs(max_workers=args.cores)