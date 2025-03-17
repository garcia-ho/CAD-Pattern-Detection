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
    Generate n visually distinct colors with high contrast.
    Uses HSV color space with well-spaced hues and alternating saturation/value patterns.
    
    Args:
        n (int): Number of colors to generate
        
    Returns:
        list: List of RGB tuples (values from 0 to 1)
    """
    colors = []
    
    # Golden ratio conjugate creates maximally separated hues
    golden_ratio_conjugate = 0.618033988749895
    
    # Start with a random offset for better color distribution
    h = random.random()
    
    # Create patterns for saturation and value to enhance contrast
    saturations = [1.0, 0.7, 1.0, 0.7, 0.9]
    values = [1.0, 1.0, 0.7, 0.9, 0.8]
    
    for i in range(n):
        # Create well-spaced hues using the golden ratio conjugate
        h = (h + golden_ratio_conjugate) % 1.0
        
        # Alternate between high saturation/value combinations for better contrast
        s = saturations[i % len(saturations)]
        v = values[i % len(values)]
        
        # Convert HSV to RGB
        rgb = colorsys.hsv_to_rgb(h, s, v)
        colors.append(rgb)
    
    # Randomize the order of colors to avoid adjacent similar colors
    random.shuffle(colors)
    
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
    Simplified approach with focus on sensitive detection.
    
    Args:
        args (tuple): (doc_path, page_num, target_strings, dpi)
    
    Returns:
        dict: Contains page number, found boxes, and processing information
    """
    doc_path, page_num, target_strings, dpi = args
    
    try:
        # Open the document
        doc = fitz.open(doc_path)
        page = doc[page_num]
        
        # Convert page to image
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72), alpha=False, colorspace="gray")
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
        
        # SIMPLIFIED PREPROCESSING
        
        # 1. Standard normalization and thresholding
        img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        binary1 = cv2.adaptiveThreshold(img_norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 7)
        
        # 2. Enhanced processing for small text
        kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        img_sharp = cv2.filter2D(img_norm, -1, kernel_sharpen)
        _, binary2 = cv2.threshold(img_sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Convert to PIL format for OCR
        pil_imgs = [
            Image.fromarray(binary1),  # Standard processing
            Image.fromarray(binary2)   # Enhanced for small text
        ]
        
        # Simplified tesseract configuration - just two effective modes
        configs = [
            # Standard configuration - good balance of accuracy and recall
            '--oem 3 --psm 6 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -."',
            
            # Sparse text mode - better for detecting isolated text in drawings
            '--oem 3 --psm 11 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -."'
        ]
        
        # Process with OCR - simplified approach
        all_boxes = []
        
        # Convert target strings to uppercase for matching
        target_strings_upper = [t.strip().upper() for t in target_strings]
        
        # Process all images with both configurations
        for img_idx, pil_img in enumerate(pil_imgs):
            for config_idx, config in enumerate(configs):
                try:
                    # Get OCR data with bounding boxes
                    ocr_data = pytesseract.image_to_data(pil_img, config=config, output_type=pytesseract.Output.DICT)
                    
                    # Process OCR results
                    for i, text in enumerate(ocr_data['text']):
                        # Skip empty text
                        if not text.strip():
                            continue
                        
                        # Accept very low confidence results to maximize recall
                        # For black and white technical drawings, even low confidence can be useful
                        if ocr_data['conf'][i] < 0:  # Only skip negative confidence
                            continue
                        
                        # Get the bounding box coordinates
                        x = ocr_data['left'][i]
                        y = ocr_data['top'][i]
                        w = ocr_data['width'][i]
                        h = ocr_data['height'][i]
                        conf = ocr_data['conf'][i]
                        
                        # Normalize text for matching
                        normalized_text = text.strip().upper()
                        
                        # SIMPLIFIED MATCHING STRATEGY - JUST THREE APPROACHES
                        for target_idx, target_upper in enumerate(target_strings_upper):
                            target_string = target_strings[target_idx]  # Original case
                            found_match = False
                            match_type = ""
                            
                            # 1. EXACT MATCH - target equals the text
                            if target_upper == normalized_text:
                                found_match = True
                                match_type = "exact"
                                
                            # 2. CONTAINED MATCH - target appears in the text
                            # This catches cases where the target is part of a longer string
                            elif target_upper in normalized_text:
                                found_match = True
                                match_type = "contained"
                            
                            # 3. WORD MATCH - target appears as a word in text
                            # This helps with targets that might be part of multi-word text
                            elif target_upper in normalized_text.split():
                                found_match = True
                                match_type = "word"
                            
                            # Add the match if found
                            if found_match:
                                all_boxes.append({
                                    'x': x, 'y': y, 'w': w, 'h': h,
                                    'text': text, 'conf': conf,
                                    'target': target_string,
                                    'source': f"img{img_idx}-config{config_idx}",
                                    'match_type': match_type
                                })
                
                except Exception as e:
                    print(f"OCR error with configuration {config_idx} on image {img_idx} on page {page_num+1}: {e}")
        
        # SIMPLIFIED DEDUPLICATION
        merged_boxes = []
        
        # Group boxes by target string
        if all_boxes:
            boxes_by_target = {}
            for box in all_boxes:
                target = box['target']
                if target not in boxes_by_target:
                    boxes_by_target[target] = []
                boxes_by_target[target].append(box)
            
            # Process each target string separately
            for target, boxes in boxes_by_target.items():
                # Sort by confidence
                boxes = sorted(boxes, key=lambda box: box['conf'], reverse=True)
                
                # Simple overlap-based deduplication
                target_merged_boxes = []
                for box in boxes:
                    # Check if this box overlaps with any existing box
                    is_new = True
                    box_rect = (box['x'], box['y'], box['x'] + box['w'], box['y'] + box['h'])
                    
                    for merged_box in target_merged_boxes:
                        merged_rect = (merged_box['x'], merged_box['y'], 
                                     merged_box['x'] + merged_box['w'], 
                                     merged_box['y'] + merged_box['h'])
                        
                        # Calculate overlap
                        x1 = max(box_rect[0], merged_rect[0])
                        y1 = max(box_rect[1], merged_rect[1])
                        x2 = min(box_rect[2], merged_rect[2])
                        y2 = min(box_rect[3], merged_rect[3])
                        
                        # If boxes overlap
                        if x1 < x2 and y1 < y2:
                            overlap_area = (x2 - x1) * (y2 - y1)
                            box_area = box['w'] * box['h']
                            merged_area = merged_box['w'] * merged_box['h']
                            
                            # If significant overlap, consider it the same match
                            if overlap_area > 0.25 * min(box_area, merged_area):
                                is_new = False
                                
                                # Keep the higher confidence match
                                if box['conf'] > merged_box['conf']:
                                    merged_box.update(box)
                                break
                    
                    # Add new box if not overlapping
                    if is_new:
                        target_merged_boxes.append(box)
                
                # Add all merged boxes for this target
                merged_boxes.extend(target_merged_boxes)
        
        # Close the document
        doc.close()
        
        return {
            'page_num': page_num,
            'boxes': merged_boxes,
            'scale': 72 / dpi
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
    Simplified approach for better detection of small text.
    
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
    
    # Prepare tasks for processing
    tasks = [(input_pdf, i, target_strings, dpi) for i in range(num_pages)]
    
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
        
        # Add highlights to PDF
        for box in boxes:
            target = box['target']
            match_type = box.get('match_type', 'unknown')
            
            # Calculate highlight box with minimal padding
            # Minimal padding to keep highlights precise
            padding = max(1, int(box['h'] * 0.1))  # Just 10% of text height
            
            x = max(0, box['x'] - padding)
            y = max(0, box['y'] - padding)
            w = box['w'] + padding * 2
            h = box['h'] + padding * 2
            
            # Convert OCR coordinates to PDF coordinates
            pdf_x = x * scale
            pdf_y = y * scale
            pdf_w = w * scale
            pdf_h = h * scale
            
            # Get color for this target - use more contrasting colors
            r, g, b = color_map.get(target, (1, 0, 0))  # Default to red if not found
            
            # Create rectangle annotation with fill
            rect = fitz.Rect(pdf_x, pdf_y, pdf_x + pdf_w, pdf_y + pdf_h)
            highlight = page.add_rect_annot(rect)
            
            # MORE TRANSPARENT FILL but STRONGER BORDER for contrast
            highlight.set_colors(stroke=(r, g, b), fill=(r, g, b, 0.05))  # Very transparent fill
            
            # Use dashed border for better visibility when colors are similar
            border_style_options = ["[1 1]", "[2 2]", "[3 3]", "[4 4]", "[1 3]", "[3 1]", "[]"]
            border_style = border_style_options[hash(target) % len(border_style_options)]  # Unique dashing per target
            highlight.set_border(width=0.5, dashes=border_style)  # Thicker border with dash pattern
            
            # Set different opacity based on match type
            match_opacities = {
                "exact": 0.7,    
                "contained": 0.5, 
                "word": 0.6,     
                "unknown": 0.4   
            }
            highlight.set_opacity(match_opacities.get(match_type, 0.5))
            
            # Add a tooltip showing the target
            highlight.set_info({"title": target})
            
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
            "recommended_cores": min(2, total_cores)  # Use at most 16 cores by default
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
    parser.add_argument("--cores", type=int, default=4, 
                        help="Number of CPU cores to use (default: 16, 0 = disable parallel processing)")
    parser.add_argument("--dpi", type=int, default=400, 
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