import os
import argparse
import re
from pathlib import Path
import fitz  # PyMuPDF for PDF handling
import cv2
import pytesseract
import numpy as np
from PIL import Image
import sys
import concurrent.futures
import multiprocessing
import csv
import random
import colorsys

def load_target_strings(target_file="target.txt"):
    """
    Load target strings from a file.
    
    Args:
        target_file (str): Path to the file containing target strings.
        
    Returns:
        list: List of target strings.
    """
    try:
        with open(target_file, 'r', encoding='utf-8') as f:
            # Read lines and strip whitespace
            targets = [line.strip() for line in f.readlines() if line.strip()]
        print(f"Loaded {len(targets)} target strings from {target_file}")
        return targets
    except Exception as e:
        print(f"Error loading target strings: {e}")
        return ["AHU"]  # Default fallback

def generate_distinct_colors(n):
    """
    Generate n visually distinct colors.
    
    Args:
        n (int): Number of colors to generate
        
    Returns:
        list: List of RGB tuples (values from 0 to 1)
    """
    colors = []
    for i in range(n):
        # Use HSV color space for better distinction
        # Vary the hue across the spectrum, keep saturation and value high
        h = i / n
        s = 0.8  # High saturation
        v = 0.9  # High value
        
        # Convert HSV to RGB
        rgb = colorsys.hsv_to_rgb(h, s, v)
        colors.append(rgb)
    
    return colors

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
    Process a single PDF page to find occurrences of target strings.
    
    Args:
        args (tuple): (doc_path, page_num, target_strings, dpi, color_map)
    
    Returns:
        dict: Contains page number, found boxes, and processing information
    """
    doc_path, page_num, target_strings, dpi, color_map = args
    
    try:
        # Open the document (each process needs its own handle)
        doc = fitz.open(doc_path)
        page = doc[page_num]
        
        print(f"Processing page {page_num + 1}/{len(doc)}")
        
        # Convert page to high-resolution image for OCR
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72), alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        
        # Convert to RGB if needed
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        # IMAGE PREPROCESSING PIPELINE
        
        # 1. Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # 2. Store original grayscale for processing
        gray_orig = gray.copy()
        
        # 3. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_clahe = clahe.apply(gray)
        
        # 4. Apply adaptive thresholding with different parameters
        binary1 = cv2.adaptiveThreshold(gray_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 15, 9)
        
        binary2 = cv2.adaptiveThreshold(gray_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 9, 3)
        
        # 5. Process images to remove lines and curves near text
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        
        horizontal_lines = cv2.morphologyEx(255-binary1, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        vertical_lines = cv2.morphologyEx(255-binary1, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        
        lines = cv2.add(horizontal_lines, vertical_lines)
        no_lines = cv2.subtract(255-binary1, lines)
        
        # 6. Create multiple processed images for better OCR accuracy
        kernel1 = np.ones((2, 2), np.uint8)
        proc_img1 = cv2.morphologyEx(binary1, cv2.MORPH_CLOSE, kernel1)
        
        kernel2 = np.ones((1, 1), np.uint8)
        proc_img2 = cv2.morphologyEx(binary2, cv2.MORPH_OPEN, kernel2)
        
        proc_img3 = 255 - no_lines
        
        # 7. Convert all processed images to PIL format
        pil_img1 = Image.fromarray(proc_img1)
        pil_img2 = Image.fromarray(proc_img2)
        pil_img3 = Image.fromarray(proc_img3)
        pil_img_orig = Image.fromarray(gray_orig)
        
        # Configure tesseract for optimal text detection
        configs = [
            '--oem 3 --psm 11 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -."',
            '--oem 3 --psm 6 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -."',
            '--oem 3 --psm 3 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -."'
        ]
        
        # Process with multiple configurations and images
        all_boxes = []
        
        for img_idx, pil_img in enumerate([pil_img1, pil_img2, pil_img3, pil_img_orig]):
            for config_idx, config in enumerate(configs):
                # Skip some combinations to reduce processing time
                if img_idx == 3 and config_idx > 1:
                    continue
                
                try:
                    # Get OCR data with bounding boxes
                    ocr_data = pytesseract.image_to_data(pil_img, config=config, output_type=pytesseract.Output.DICT)
                    
                    # Process OCR results
                    for i, text in enumerate(ocr_data['text']):
                        # Skip empty text or very low confidence
                        if not text.strip() or ocr_data['conf'][i] < 0:
                            continue
                        
                        # Check each target string
                        normalized_text = text.strip().upper()
                        
                        for target_string in target_strings:
                            target_upper = target_string.upper()
                            
                            # Check if target string is in the text
                            if target_upper in normalized_text:
                                # Get the bounding box coordinates
                                x = ocr_data['left'][i]
                                y = ocr_data['top'][i]
                                w = ocr_data['width'][i]
                                h = ocr_data['height'][i]
                                conf = ocr_data['conf'][i]
                                
                                # Add box info with target string info for highlighting
                                all_boxes.append({
                                    'x': x, 'y': y, 'w': w, 'h': h, 
                                    'text': text, 'conf': conf,
                                    'target': target_string,
                                    'source': f"img{img_idx}-config{config_idx}"
                                })
                
                except Exception as e:
                    print(f"OCR error with configuration {config_idx} on image {img_idx} on page {page_num+1}: {e}")
        
        # Deduplicate boxes for each target string
        merged_boxes = []
        
        if all_boxes:
            # Group boxes by target string
            boxes_by_target = {}
            for box in all_boxes:
                target = box['target']
                if target not in boxes_by_target:
                    boxes_by_target[target] = []
                boxes_by_target[target].append(box)
            
            # Process each target string separately for deduplication
            for target, boxes in boxes_by_target.items():
                # Sort by confidence
                boxes = sorted(boxes, key=lambda box: box['conf'], reverse=True)
                
                target_merged_boxes = []
                for box in boxes:
                    # Check if this box overlaps with any already merged box
                    is_new = True
                    box_rect = (box['x'], box['y'], box['x'] + box['w'], box['y'] + box['h'])
                    
                    for idx, merged_box in enumerate(target_merged_boxes):
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
                                
                                # Expand to union
                                merged_box['x'] = min(box_rect[0], merged_rect[0])
                                merged_box['y'] = min(box_rect[1], merged_rect[1])
                                merged_box['w'] = max(box_rect[2], merged_rect[2]) - merged_box['x']
                                merged_box['h'] = max(box_rect[3], merged_rect[3]) - merged_box['y']
                                break
                    
                    # Add new box if not merged
                    if is_new:
                        target_merged_boxes.append(box)
                
                # Add all merged boxes for this target to the final list
                merged_boxes.extend(target_merged_boxes)
        
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

def highlight_strings_in_pdf(input_pdf, output_pdf, target_strings, max_workers=16, dpi=300):
    """
    Find and highlight occurrences of multiple strings in PDF with colored boxes.
    Enhanced with better handling of special cases.
    
    Args:
        input_pdf (str): Path to input PDF
        output_pdf (str): Path to save highlighted PDF
        target_strings (list): List of strings to highlight
        max_workers (int): Maximum number of parallel workers (0 = disable parallel processing)
        dpi (int): DPI resolution for OCR
    
    Returns:
        dict: Count of occurrences for each target string
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    
    # Generate distinct colors for each target string
    colors = generate_distinct_colors(len(target_strings))
    color_map = {target: color for target, color in zip(target_strings, colors)}
    
    # Determine processing mode
    parallel_mode = max_workers > 0
    
    if parallel_mode:
        # Ensure max_workers is a reasonable number if parallel mode is on
        if max_workers is None:
            cpu_info = get_cpu_info()
            max_workers = cpu_info["recommended_cores"]
    
    # Open the PDF to get page count
    doc = fitz.open(input_pdf)
    num_pages = len(doc)
    doc.close()
    
    # Prepare tasks for processing - ADD THE COLOR_MAP PARAMETER HERE
    tasks = [(input_pdf, i, target_strings, dpi, color_map) for i in range(num_pages)]
    
    # Process pages (either in parallel or sequentially)
    results = []
    if parallel_mode:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Process tasks in batches to avoid memory issues
            batch_size = min(100, num_pages)  # Maximum 100 pages per batch
            for i in range(0, len(tasks), batch_size):
                batch_tasks = tasks[i:i+batch_size]
                batch_results = list(executor.map(process_pdf_page, batch_tasks))
                results.extend(batch_results)
                print(f"Processed pages {i+1}-{min(i+batch_size, num_pages)} of {num_pages}")
    else:
        # Sequential processing
        for i, task in enumerate(tasks):
            result = process_pdf_page(task)
            results.append(result)
            print(f"Processed page {i+1}/{num_pages}")
    
    # Open the PDF again for highlighting
    doc = fitz.open(input_pdf)
    
    # Track occurrences of each target string
    occurrences = {target: 0 for target in target_strings}
    
    # Apply highlights based on results
    for result in results:
        if not result or 'error' in result:
            continue
            
        page_num = result['page_num']
        boxes = result['boxes']
        scale = result['scale']
        
        page = doc[page_num]
        
        # Add highlights to PDF - batch process for efficiency
        for box in boxes:
            target = box['target']
            
            # Adjust padding based on special cases
            if box.get('in_margin', False):
                # Use less padding in margins to avoid crossing page boundaries
                padding = int(box['h'] * 0.15)  # 15% padding
            elif box.get('near_line', False):
                # Use more padding when near lines to ensure full coverage
                padding = int(box['h'] * 0.25)  # 25% padding
            else:
                # Standard padding
                padding = int(box['h'] * 0.2)  # 20% padding
            
            x = max(0, box['x'] - padding)
            y = max(0, box['y'] - padding)
            w = box['w'] + padding * 2
            h = box['h'] + padding * 2
            
            # Convert OCR coordinates to PDF coordinates
            pdf_x = x * scale
            pdf_y = y * scale
            pdf_w = w * scale
            pdf_h = h * scale
            
            # Get color for this target
            r, g, b = color_map.get(target, (1, 0, 0))  # Default to red if not found
            
            # Adjust opacity based on match confidence/score
            base_opacity = 0.4
            if 'score' in box:
                if box['score'] < 80:  # Lower scoring matches (fuzzy/partial)
                    opacity = max(0.3, base_opacity * (box['score'] / 100))
                    stroke_width = 0.7  # Thinner borders for less confident matches
                else:  # Higher scoring matches
                    opacity = base_opacity
                    stroke_width = 1.0
            else:
                opacity = base_opacity
                stroke_width = 1.0
            
            # Create rectangle annotation with fill
            rect = fitz.Rect(pdf_x, pdf_y, pdf_x + pdf_w, pdf_y + pdf_h)
            highlight = page.add_rect_annot(rect)
            highlight.set_colors(stroke=(r, g, b), fill=(r, g, b, 0.2))
            highlight.set_opacity(opacity)
            
            # Set border width
            highlight.set_border(width=stroke_width)
            
            highlight.update()
            
            # Increment occurrence count
            occurrences[target] += 1
    
    # Save the highlighted PDF
    doc.save(output_pdf)
    doc.close()
    
    return occurrences

def get_cpu_info():
    """Get information about available CPU resources"""
    try:
        total_cores = multiprocessing.cpu_count()
        return {
            "total_cores": total_cores,
            "recommended_cores": min(16, total_cores)  # Use at most 16 cores by default
        }
    except:
        return {"total_cores": "Unknown", "recommended_cores": 4}

def process_all_pdfs(input_folder="String_match/input_pdf", output_folder="String_match/output", 
                   target_file="target.txt", max_workers=16, dpi=300):
    """
    Process all PDFs in the input folder, find target string occurrences and highlight them.
    Output results to a CSV file.
    
    Args:
        input_folder (str): Path to folder containing PDFs
        output_folder (str): Path to save highlighted PDFs and report
        target_file (str): Path to file containing target strings
        max_workers (int): Maximum number of workers per PDF (0 = disable parallel processing)
        dpi (int): DPI resolution for OCR
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Load target strings
    target_strings = load_target_strings(target_file)
    if not target_strings:
        print("No target strings found. Please check your target.txt file.")
        return
    
    # Create a log file to store all processing details
    log_path = os.path.join(output_folder, "processing_log.txt")
    log_file = open(log_path, 'w', encoding='utf-8')
    
    # Write header to log file
    log_file.write("Target String Detection Processing Log\n")
    log_file.write("====================================\n\n")
    log_file.write(f"Target Strings: {', '.join(target_strings)}\n\n")
    
    # Display CPU information
    cpu_info = get_cpu_info()
    total_cores = cpu_info["total_cores"]
    
    log_file.write("System Information:\n")
    log_file.write(f"- Total CPU Cores: {total_cores}\n")
    
    # Determine processing mode
    parallel_mode = max_workers > 0
    if parallel_mode:
        log_file.write(f"- Using {max_workers} cores for processing\n")
    else:
        log_file.write("- Parallel processing disabled. Using single-core sequential processing.\n")
    
    # For the console, we'll display minimal information
    print(f"Processing PDFs with {'parallel' if parallel_mode else 'sequential'} mode...")
    print(f"Looking for {len(target_strings)} target strings: {', '.join(target_strings)}")
    
    # Get list of PDFs to process
    pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {input_folder}")
        log_file.write("No PDF files found.\n")
        log_file.close()
        return
    
    log_file.write(f"\nFound {len(pdf_files)} PDF files to process\n\n")
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Prepare CSV file for results
    csv_path = os.path.join(output_folder, "detection_results.csv")
    
    # Create CSV header row with target strings
    csv_header = ["Filename"] + target_strings + ["Total"]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(csv_header)
        
        # Process each PDF in the input folder
        for filename in pdf_files:
            input_path = os.path.join(input_folder, filename)
            base_name = os.path.splitext(filename)[0]
            highlighted_pdf_path = os.path.join(output_folder, f"{base_name}_highlighted.pdf")
            
            print(f"Processing {filename}... ", end='', flush=True)
            log_file.write(f"Processing {filename}:\n")
            log_file.write(f"- Input: {input_path}\n")
            log_file.write(f"- Output: {highlighted_pdf_path}\n")
            
            # Temporarily redirect stdout to log file
            original_stdout = sys.stdout
            sys.stdout = log_file
            
            # Find and highlight target string occurrences
            occurrences = highlight_strings_in_pdf(
                input_path, highlighted_pdf_path, 
                target_strings, max_workers=max_workers, 
                dpi=dpi
            )
            
            # Restore stdout
            sys.stdout = original_stdout
            
            # Calculate total occurrences
            total_count = sum(occurrences.values())
            
            # Print minimal info to console
            print(f"found {total_count} total occurrences")
            
            # Write results to CSV
            csv_row = [filename]
            for target in target_strings:
                csv_row.append(occurrences.get(target, 0))
            csv_row.append(total_count)
            
            csv_writer.writerow(csv_row)
            
            # Log occurrences to log file
            log_file.write("- Occurrences:\n")
            for target, count in occurrences.items():
                log_file.write(f"  - '{target}': {count}\n")
            log_file.write(f"- Total: {total_count}\n\n")
    
    log_file.write("\nProcessing completed.\n")
    log_file.close()
    
    print(f"\nProcessing complete!")
    print(f"- Total files processed: {len(pdf_files)}")
    print(f"- Results saved to: {csv_path}")
    print(f"- Detailed log saved to: {log_path}")


if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Find and highlight target strings in PDFs using OCR")
    parser.add_argument("pdf_file", nargs="?", help="Optional: specific PDF file to process")
    parser.add_argument("--targets", default="String_match/target.txt", help="File containing target strings (one per line)")
    parser.add_argument("--cores", type=int, default=16, 
                        help="Number of CPU cores to use (default: 16, 0 = disable parallel processing)")
    parser.add_argument("--dpi", type=int, default=300, 
                        help="DPI resolution for OCR (higher = more accurate but slower, default: 300)")

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
            
            # Load target strings
            target_strings = load_target_strings(args.targets)
            
            # Process the file - directly use args.dpi
            highlight_strings_in_pdf(input_path, output_pdf, target_strings, max_workers=args.cores, dpi=args.dpi)
    else:
        # Process all PDFs in the folder - directly use args.dpi
        process_all_pdfs(target_file=args.targets, max_workers=args.cores, dpi=args.dpi)