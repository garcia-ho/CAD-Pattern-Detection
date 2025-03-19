import os
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
import time
import pdfplumber

from preprocessing import process_pdf_page


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
    if (output_txt_path is None):
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



def highlight_strings_in_pdf(input_pdf, output_pdf, target_strings, max_workers=4, dpi=300, sensitivity=5):
    """
    Highlight occurrences of target strings in a PDF and save the result.
    Uses both custom OCR and embedded text extraction for maximum detection.
    
    Args:
        input_pdf (str): Path to the input PDF file
        output_pdf (str): Path to save the highlighted PDF
        target_strings (list): List of target strings to find and highlight
        max_workers (int): Maximum number of parallel workers (0 = disable parallel processing)
        dpi (int): DPI resolution for OCR
        sensitivity (int): Detection sensitivity (1-10, where 1 is strict and 10 is very permissive)
    
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
    
    # PART 1: Use custom OCR to find matches
    # ---------------------------------------
    
    # Prepare tasks for processing - include sensitivity parameter
    tasks = [(input_pdf, i, target_strings, dpi, sensitivity) for i in range(num_pages)]
    
    # Process pages (either in parallel or sequentially)
    ocr_results = []
    if (parallel_mode):
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Process tasks in batches to avoid memory issues
            batch_size = min(100, num_pages)  # Maximum 100 pages per batch
            for i in range(0, len(tasks), batch_size):
                batch_tasks = tasks[i:i+batch_size]
                batch_results = list(executor.map(process_pdf_page, batch_tasks))
                ocr_results.extend(batch_results)
                print(f"Processed pages {i+1}-{min(i+batch_size, num_pages)} of {num_pages}")
    else:
        # Sequential processing
        for i, task in enumerate(tasks):
            result = process_pdf_page(task)
            ocr_results.append(result)
            print(f"Processed page {i+1}/{num_pages}")
    
    # PART 2: Extract text with positions from embedded text layer
    # -----------------------------------------------------------
    print("Extracting embedded text from PDF...")
    text_elements = extract_text_with_positions(input_pdf)
    
    # Process text elements to find matches
    embedded_matches = []
    
    print("Finding matches in embedded text...")
    for target in target_strings:
        target_upper = target.upper()
        for element in text_elements:
            text = element['text'].upper()
            
            # Check for matches (exact, word, or contained)
            if (target_upper == text or                   # exact match
                target_upper in text.split() or          # word match
                target_upper in text):                   # contained match
                
                # Create a match object similar to OCR boxes
                match_type = "exact" if target_upper == text else "word" if target_upper in text.split() else "contained"
                
                embedded_matches.append({
                    'page_num': element['page_num'],
                    'x': element['x0'],
                    'y': element['y0'],
                    'w': element['width'],
                    'h': element['height'],
                    'text': element['text'],
                    'conf': element.get('confidence', 90),
                    'target': target,
                    'source': element['source'],
                    'match_type': match_type
                })
    
    print(f"Found {len(embedded_matches)} matches in embedded text")
    
    # PART 3: Combine results and highlight in PDF
    # -------------------------------------------
    
    # Open the PDF again for highlighting
    doc = fitz.open(input_pdf)
    
    # Track occurrences of each target string
    occurrences = {target: 0 for target in target_strings}
    
    # Apply highlights from custom OCR results
    for result in ocr_results:
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
    
    # Apply highlights from embedded text matches
    for box in embedded_matches:
        page_num = box['page_num']
        page = doc[page_num]
        target = box['target']
        match_type = box.get('match_type', 'unknown')
        
        # Check for overlap with existing annotations to avoid duplicates
        new_rect = fitz.Rect(box['x'], box['y'], box['x'] + box['w'], box['y'] + box['h'])
        
        # Check if this area overlaps with existing annotations
        overlap = False
        for annot in page.annots():
            if annot.rect.intersect(new_rect).get_area() > 0.3 * min(annot.rect.get_area(), new_rect.get_area()):
                overlap = True
                break
        
        if overlap:
            continue  # Skip this match as it overlaps with existing annotation
        
        # Get color for this target - use more contrasting colors
        r, g, b = color_map.get(target, (1, 0, 0))
        
        # Create rectangle annotation with fill
        highlight = page.add_rect_annot(new_rect)
        
        # More transparent fill with strong border for embedded text
        highlight.set_colors(stroke=(r, g, b), fill=(r, g, b, 0.05))
        
        # Different border style for embedded text matches
        highlight.set_border(width=0.5, dashes="[1 3]")
        
        # Set opacity based on match type
        match_opacities = {
            "exact": 0.7,
            "contained": 0.5,
            "word": 0.6,
            "unknown": 0.4
        }
        highlight.set_opacity(match_opacities.get(match_type, 0.5))
        
        # Add tooltip showing it's from embedded text
        highlight.set_info({"title": f"{target} (embedded text)"})
        
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
            "recommended_cores": min(4, total_cores)  # Use at most 16 cores by default
        }
    except:
        return {"total_cores": "Unknown", "recommended_cores": 4}


def process_all_pdfs(input_folder="String_match/input_pdf", output_folder="String_match/output", 
                   target_file="target.txt", max_workers=4, dpi=300, sensitivity=7):
    """
    Process all PDFs in the input folder, find target string occurrences and highlight them.
    Output results to a CSV file with timing information.
    
    Args:
        input_folder (str): Path to folder containing PDFs
        output_folder (str): Path to save highlighted PDFs and report
        target_file (str): Path to file containing target strings
        max_workers (int): Maximum number of workers per PDF (0 = disable parallel processing)
        dpi (int): DPI resolution for OCR
        sensitivity (int): Detection sensitivity (1-10, where 1 is strict and 10 is very permissive)
    """
    # Start overall processing time
    overall_start_time = time.time()
    
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
    csv_header = ["Filename"] + target_strings + ["Total", "Processing Time (sec)"]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(csv_header)
        
        # Track total processing time
        processing_times = []
        
        # Process each PDF in the input folder
        for filename in pdf_files:
            input_path = os.path.join(input_folder, filename)
            base_name = os.path.splitext(filename)[0]
            highlighted_pdf_path = os.path.join(output_folder, f"{base_name}_highlighted.pdf")
            
            print(f"Processing {filename}... ", end='', flush=True)
            log_file.write(f"Processing {filename}:\n")
            log_file.write(f"- Input: {input_path}\n")
            log_file.write(f"- Output: {highlighted_pdf_path}\n")
            
            # Start time for this PDF
            start_time = time.time()
            
            # Temporarily redirect stdout to log file
            original_stdout = sys.stdout
            sys.stdout = log_file
            
            # Find and highlight target string occurrences
            occurrences = highlight_strings_in_pdf(
                input_path, highlighted_pdf_path, 
                target_strings, max_workers=max_workers, 
                dpi=dpi, sensitivity=sensitivity
            )
            
            # End time for this PDF
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Restore stdout
            sys.stdout = original_stdout
            
            # Calculate total occurrences
            total_count = sum(occurrences.values())
            
            # Print minimal info to console
            print(f"found {total_count} total occurrences in {processing_time:.2f} seconds")
            
            # Write results to CSV
            csv_row = [filename]
            for target in target_strings:
                csv_row.append(occurrences.get(target, 0))
            csv_row.append(total_count)
            csv_row.append(f"{processing_time:.2f}")
            
            csv_writer.writerow(csv_row)
            
            # Log occurrences and timing to log file
            log_file.write("- Occurrences:\n")
            for target, count in occurrences.items():
                log_file.write(f"  - '{target}': {count}\n")
            log_file.write(f"- Total: {total_count}\n")
            log_file.write(f"- Processing Time: {processing_time:.2f} seconds\n\n")
    
    # Calculate overall statistics
    overall_time = time.time() - overall_start_time
    avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
    
    # Write summary statistics
    log_file.write("\nProcessing Summary:\n")
    log_file.write(f"- Total files processed: {len(pdf_files)}\n")
    log_file.write(f"- Total processing time: {overall_time:.2f} seconds\n")
    log_file.write(f"- Average time per PDF: {avg_time:.2f} seconds\n")
    if processing_times:
        log_file.write(f"- Fastest PDF: {min(processing_times):.2f} seconds\n")
        log_file.write(f"- Slowest PDF: {max(processing_times):.2f} seconds\n")
    log_file.write("\nProcessing completed.\n")
    log_file.close()
    
    print(f"\nProcessing complete!")
    print(f"- Total files processed: {len(pdf_files)}")
    print(f"- Total processing time: {overall_time:.2f} seconds")
    print(f"- Results saved to: {csv_path}")
    print(f"- Detailed log saved to: {log_path}")

def extract_text_with_positions(pdf_path):
    """
    Extract text with position information from PDF using pdfplumber for embedded text
    and custom OCR for image-based text. Prioritizes custom OCR with embedded text as supplement.
    
    Args:
        pdf_path (str): Path to PDF file
        
    Returns:
        list: List of dictionaries with text, page_num, and position information
    """
    text_elements = []
    
    try:
        # First extract text with pdfplumber (embedded text)
        embedded_elements = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Extract text with positions
                words = page.extract_words(x_tolerance=3, y_tolerance=3, keep_blank_chars=False)
                
                for word in words:
                    embedded_elements.append({
                        'page_num': page_num,
                        'text': word['text'],
                        'x0': word['x0'],
                        'y0': word['top'],
                        'x1': word['x1'],
                        'y1': word['bottom'],
                        'width': word['x1'] - word['x0'],
                        'height': word['bottom'] - word['top'],
                        'source': 'embedded',
                        'confidence': 90  # Generally high confidence for embedded text
                    })
                    
        print(f"Extracted {len(embedded_elements)} embedded text elements from {pdf_path}")
        
        # Open the document with PyMuPDF for OCR processing
        doc = fitz.open(pdf_path)
        
        # Process each page with custom OCR
        ocr_elements = []
        for page_num, page in enumerate(doc):
            # Get a higher resolution image for better OCR accuracy
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72), alpha=False)
            
            # Convert the pixmap data to a numpy array
            if pix.n == 1:  # Grayscale
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
                img_rgb = np.stack([img, img, img], axis=2)
            elif pix.n == 3:  # RGB
                img_rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            elif pix.n == 4:  # RGBA
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 4)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                
            # Convert to PIL for OCR
            pil_img = Image.fromarray(img_rgb)
            
            # Use pytesseract to get text with positions
            ocr_data = pytesseract.image_to_data(pil_img, config='--psm 11', output_type=pytesseract.Output.DICT)
            
            scale_factor = 72 / 300  # Convert from OCR coordinates to PDF coordinates
            
            # Process OCR results
            for i, text in enumerate(ocr_data['text']):
                # Skip empty results
                if not text.strip():
                    continue
                
                # Get confidence score
                conf = ocr_data['conf'][i]
                if conf < 0:  # Skip negative confidence scores
                    continue
                    
                # Get bounding box
                x = ocr_data['left'][i] * scale_factor
                y = ocr_data['top'][i] * scale_factor
                w = ocr_data['width'][i] * scale_factor
                h = ocr_data['height'][i] * scale_factor
                
                ocr_elements.append({
                    'page_num': page_num,
                    'text': text,
                    'x0': x,
                    'y0': y,
                    'x1': x + w,
                    'y1': y + h,
                    'width': w,
                    'height': h,
                    'source': 'custom_ocr',
                    'confidence': conf
                })
        
        doc.close()
        print(f"Extracted {len(ocr_elements)} custom OCR text elements from {pdf_path}")
        
        # Merge results, prioritizing custom OCR but including all elements
        # First add all OCR elements (prioritized)
        text_elements.extend(ocr_elements)
        
        # Then add embedded elements that don't overlap significantly with OCR elements
        for emb in embedded_elements:
            overlap = False
            for ocr in ocr_elements:
                # Skip if not on same page
                if emb['page_num'] != ocr['page_num']:
                    continue
                    
                # Check for significant overlap
                x_overlap = max(0, min(emb['x1'], ocr['x1']) - max(emb['x0'], ocr['x0']))
                y_overlap = max(0, min(emb['y1'], ocr['y1']) - max(emb['y0'], ocr['y0']))
                
                if x_overlap > 0 and y_overlap > 0:
                    area_overlap = x_overlap * y_overlap
                    area_emb = emb['width'] * emb['height']
                    area_ocr = ocr['width'] * ocr['height']
                    
                    # If overlap is significant (>30% of either box)
                    if area_overlap > 0.3 * min(area_emb, area_ocr):
                        overlap = True
                        break
            
            # If no significant overlap with OCR elements, add it
            if not overlap:
                text_elements.append(emb)
            
        print(f"Total {len(text_elements)} text elements after merging OCR and embedded text")
        return text_elements
    except Exception as e:
        print(f"Error extracting text with positions: {e}")
        return []
