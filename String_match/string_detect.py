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
import time
import pdfplumber
import pdf2image
import pdfminer
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar, LTTextLine
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams

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

def process_pdf_page(args):
    """
    Process a single PDF page to find occurrences of target strings using OCR only.
    Aggressively filters out grey lines/text (RGB >= 128,128,128) and focuses on detecting 
    only black text (RGB close to 0,0,0) on white background, including text partially 
    obscured by lines.
    
    Args:
        args (tuple): (doc_path, page_num, target_strings, dpi, sensitivity)
    
    Returns:
        dict: Contains page number, found boxes, and processing information
    """
    if len(args) == 5:
        doc_path, page_num, target_strings, dpi, sensitivity = args
    else:
        # Default sensitivity for backward compatibility
        doc_path, page_num, target_strings, dpi = args
        sensitivity = 5
    
    # Apply higher sensitivity for more aggressive detection
    sensitivity = max(5, min(10, sensitivity))
    
    try:
        # Initialize all_boxes to store matched text
        all_boxes = []
        
        # Extract filename from path for output images
        base_filename = os.path.splitext(os.path.basename(doc_path))[0]
        
        # Open the document with PyMuPDF for page rendering only
        doc = fitz.open(doc_path)
        page = doc[page_num]
        
        # Convert target strings to uppercase for case-insensitive matching
        target_strings_upper = [t.strip().upper() for t in target_strings]
        
        # Create variants for OCR error handling
        target_variants = {}
        for i, target in enumerate(target_strings_upper):
            variants = [target]  # Original target
            
            # Add common OCR errors
            if 'O' in target:
                variants.append(target.replace('O', '0'))
            if '0' in target:
                variants.append(target.replace('0', 'O'))
            if 'I' in target:
                variants.append(target.replace('I', '1'))
            if '1' in target:
                variants.append(target.replace('1', 'I'))
            if 'S' in target:
                variants.append(target.replace('S', '5'))
            if '5' in target:
                variants.append(target.replace('5', 'S'))
            if 'Z' in target:
                variants.append(target.replace('Z', '2'))
            if '2' in target:
                variants.append(target.replace('2', 'Z'))
            if 'B' in target:
                variants.append(target.replace('B', '8'))
            if '8' in target:
                variants.append(target.replace('8', 'B'))
                
            # Handle spaces and dashes
            if ' ' in target:
                variants.append(target.replace(' ', ''))  # Remove spaces
                variants.append(target.replace(' ', '-'))  # Space to dash
            
            if '-' in target:
                variants.append(target.replace('-', ''))  # Remove dashes
                variants.append(target.replace('-', ' '))  # Dash to space
                
            # Store all unique variants
            target_variants[i] = list(set(variants))
        
        # ENHANCED PREPROCESSING: FILTER GREY, REMOVE LINES, APPLY NOISE REDUCTION
        print(f"Page {page_num+1}: Filtering grey elements and cleaning with noise reduction...")

        # Get the page dimensions from the PDF directly
        page_width = page.rect.width
        page_height = page.rect.height

        # Render the page with RGB color
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72), alpha=False)

        # Convert the pixmap data to a numpy array
        if pix.n == 1:  # Grayscale
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
            img_rgb = np.stack([img, img, img], axis=2)
        elif pix.n == 3:  # RGB
            img_rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        elif pix.n == 4:  # RGBA
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 4)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        # Extract filename for output image
        base_filename = os.path.splitext(os.path.basename(doc_path))[0]

        # STEP 1: FILTER GREY - Remove all pixels with RGB values >= 128,128,128
        # Create a mask where all RGB values must be < 128 (non-grey elements)
        black_mask = np.logical_and.reduce((
            img_rgb[:,:,0] < 128,
            img_rgb[:,:,1] < 128,
            img_rgb[:,:,2] < 128
        ))

        # Create a new image with white background
        filtered_img = np.ones_like(img_rgb) * 255  # White background

        # Set the black text pixels
        filtered_img[black_mask] = [0, 0, 0]  # Black text

        # STEP 2: CONVERT TO GRAYSCALE
        gray = cv2.cvtColor(filtered_img, cv2.COLOR_RGB2GRAY)

        # STEP 3: LINE REMOVAL - Remove long horizontal and vertical lines
        # Create a working copy
        img_no_lines = gray.copy()

        # Detect and remove LONG horizontal lines only
        # Use a longer kernel to target only long horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        horizontal_lines = cv2.morphologyEx(255 - gray, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

        # Detect and remove LONG vertical lines only
        # Use a longer kernel to target only long vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
        vertical_lines = cv2.morphologyEx(255 - gray, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

        # Combine horizontal and vertical lines
        lines = cv2.add(horizontal_lines, vertical_lines)

        # Remove lines from the image (set to white)
        img_no_lines[lines > 0] = 255

        # STEP 4: APPLY NOISE REDUCTION TECHNIQUES

        # 4.1: Apply bilateral filter to reduce noise while preserving edges
        # This is good for preserving text edges while smoothing noise
        bilateral_filtered = cv2.bilateralFilter(img_no_lines, d=5, sigmaColor=75, sigmaSpace=75)

        # 4.2: Apply non-local means denoising for high-quality noise removal
        # This algorithm is particularly good for document images
        denoised = cv2.fastNlMeansDenoising(bilateral_filtered, h=10, templateWindowSize=7, searchWindowSize=21)

        # 4.3: Apply mild median blur to remove salt-and-pepper noise
        # Small kernel size (3) to avoid blurring text
        median_filtered = cv2.medianBlur(denoised, 3)

        # STEP 5: CLEAN UP THE IMAGE
        # Apply very light morphological opening to remove small noise
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(median_filtered, cv2.MORPH_OPEN, kernel, iterations=1)

        # STEP 6: ENHANCE CONTRAST FOR BETTER TEXT READABILITY
        # Apply contrast limited adaptive histogram equalization (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(cleaned)

        # Create output directory
        os.makedirs('String_match/filtered_img', exist_ok=True)

        # STEP 7: SAVE AS PDF
        pdf_path = f"String_match/filtered_img/{base_filename}_page{page_num+1}_filtered.pdf"

        # Ensure the output directory exists
        os.makedirs('String_match/filtered_img', exist_ok=True)

        try:
            # Method 1: Use pymupdf (fitz) to create the PDF
            output_pdf = fitz.open()
            pdf_page = output_pdf.new_page(width=page_width, height=page_height)
            
            # Save enhanced image as temporary PNG file
            temp_png_path = f"String_match/filtered_img/{base_filename}_page{page_num+1}_temp.png"
            cv2.imwrite(temp_png_path, enhanced)
            
            # Insert the image using file path (more reliable than streams)
            img_rect = fitz.Rect(0, 0, page_width, page_height)
            pdf_page.insert_image(img_rect, filename=temp_png_path)
            
            # Save and close the PDF
            output_pdf.save(pdf_path)
            output_pdf.close()
            
            # Remove temporary file
            try:
                os.remove(temp_png_path)
            except:
                pass
            
            print(f"Page {page_num+1}: Saved filtered PDF to {pdf_path}")
            
        except Exception as e:
            print(f"Error saving PDF with PyMuPDF: {e}")
            
            # Fallback method: Use PIL to save as PNG if PDF fails
            fallback_png_path = f"String_match/filtered_img/{base_filename}_page{page_num+1}_filtered.png"
            cv2.imwrite(fallback_png_path, enhanced)
            print(f"Page {page_num+1}: Saved fallback PNG to {fallback_png_path}")

        # Create a collection of processed images for OCR
        filtered_images = [
            ("processed", enhanced)
        ]

        # Process the filtered image with OCR
        for filter_name, filtered_img in filtered_images:
            # Convert to PIL for OCR
            pil_img = Image.fromarray(filtered_img)
            
            # OCR configs for different text patterns
            configs = [
                # Standard config - for normal text blocks
                '--oem 3 --psm 6 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -./()[]{}#"',
                
                # Sparse text config - for CAD drawings with isolated text
                '--oem 3 --psm 11 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -./()[]{}#"',
                
                # Single line config - for text in a single line
                '--oem 3 --psm 7 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -./()[]{}#"',
            ]
            
            # Process with each OCR config
            for config_idx, config in enumerate(configs):
                try:
                    # Get OCR data with bounding boxes
                    ocr_data = pytesseract.image_to_data(pil_img, config=config, output_type=pytesseract.Output.DICT)
                    
                    # Process OCR results
                    for i, text in enumerate(ocr_data['text']):
                        # Skip empty text
                        if not text.strip():
                            continue
                        
                        # Apply confidence threshold based on sensitivity
                        min_confidence = max(25, 50 - (sensitivity * 5))  # More permissive for higher sensitivity
                        if ocr_data['conf'][i] < min_confidence:
                            continue
                        
                        # Get bounding box
                        x = ocr_data['left'][i]
                        y = ocr_data['top'][i]
                        w = ocr_data['width'][i]
                        h = ocr_data['height'][i]
                        conf = ocr_data['conf'][i]
                        
                        # Skip extremely small or large boxes (likely noise or page boundaries)
                        if w < 3 or h < 3 or w > pil_img.width * 0.9 or h > pil_img.height * 0.9:
                            continue
                        
                        # Scale coordinates back to original PDF size
                        scale_to_pdf = 72 / dpi
                        pdf_x = x * scale_to_pdf
                        pdf_y = y * scale_to_pdf
                        pdf_w = w * scale_to_pdf
                        pdf_h = h * scale_to_pdf
                        
                        # Normalize text for matching
                        normalized_text = text.strip().upper()
                        
                        # Clean text for matching
                        cleaned_text = re.sub(r'[^\w\s\-\.]', '', normalized_text)
                        
                        # Check matches with each target
                        for target_idx, target_upper in enumerate(target_strings_upper):
                            target_string = target_strings[target_idx]
                            curr_variants = target_variants.get(target_idx, [target_upper])
                            
                            for variant in curr_variants:
                                found_match = False
                                match_type = ""
                                
                                # 1. EXACT MATCH
                                if variant == normalized_text or variant == cleaned_text:
                                    found_match = True
                                    match_type = "exact"
                                
                                # 2. WORD MATCH
                                elif variant in normalized_text.split() or variant in cleaned_text.split():
                                    found_match = True
                                    match_type = "word"
                                
                                # 3. CONTAINED MATCH
                                elif variant in normalized_text or variant in cleaned_text:
                                    found_match = True
                                    match_type = "contained"
                                
                                # If found a match, add to results
                                if found_match:
                                    all_boxes.append({
                                        'x': pdf_x, 'y': pdf_y, 'w': pdf_w, 'h': pdf_h,
                                        'text': text, 'conf': conf,
                                        'target': target_string,
                                        'source': f"{filter_name}-cfg{config_idx}",
                                        'match_type': match_type,
                                        'variant': variant
                                    })
                                    break
                except Exception as e:
                    print(f"OCR error with config {config_idx} on filter {filter_name} on page {page_num+1}: {e}")
        
        # Close the document
        doc.close()
        
        print(f"Page {page_num+1}: Found {len(all_boxes)} potential matches before deduplication")
        
        # DEDUPLICATION
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
                # Define priorities for deduplication
                match_type_priority = {
                    "exact": 3,
                    "word": 2, 
                    "contained": 1,
                    "unknown": 0
                }
                
                # Sort boxes first by location to group nearby detections
                boxes = sorted(boxes, key=lambda box: (box['y'], box['x']))
                
                # Overlap threshold for deduplication
                overlap_threshold = 0.3
                
                # Deduplication
                target_merged_boxes = []
                for box in boxes:
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
                            
                            if overlap_area > overlap_threshold * min(box_area, merged_area):
                                is_new = False
                                
                                # Compare boxes to keep the better one
                                box_priority = match_type_priority.get(box.get('match_type', 'unknown'), 0)
                                merged_priority = match_type_priority.get(merged_box.get('match_type', 'unknown'), 0)
                                
                                # Prefer higher match type priority, then higher confidence
                                if (box_priority > merged_priority) or \
                                   (box_priority == merged_priority and box['conf'] > merged_box['conf']):
                                    merged_box.update(box)
                                break
                    
                    # Add new box if not overlapping
                    if is_new:
                        target_merged_boxes.append(box)
                
                # Add all merged boxes for this target
                merged_boxes.extend(target_merged_boxes)
        
        print(f"Page {page_num+1}: Found {len(merged_boxes)} unique matches after deduplication")
        
        return {
            'page_num': page_num,
            'boxes': merged_boxes,
            'scale': 1.0
        }
        
    except Exception as e:
        print(f"Error processing page {page_num+1}: {e}")
        return {
            'page_num': page_num,
            'boxes': [],
            'error': str(e)
        }

def highlight_strings_in_pdf(input_pdf, output_pdf, target_strings, max_workers=4, dpi=300, sensitivity=5):
    """
    Highlight occurrences of target strings in a PDF and save the result.
    
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
    
    # Prepare tasks for processing - include sensitivity parameter
    tasks = [(input_pdf, i, target_strings, dpi, sensitivity) for i in range(num_pages)]
    
    # Process pages (either in parallel or sequentially)
    results = []
    if (parallel_mode):
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


if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Find and highlight target strings in PDFs using OCR")
    parser.add_argument("pdf_file", nargs="?", help="Optional: specific PDF file to process")
    parser.add_argument("--targets", default="String_match/target.txt", help="File containing target strings (one per line)")
    parser.add_argument("--cores", type=int, default=8, 
                        help="Number of CPU cores to use (default: 4, 0 = disable parallel processing)")
    parser.add_argument("--dpi", type=int, default=500, 
                        help="DPI resolution for OCR (higher = more accurate but slower, default: 400)")
    parser.add_argument("--sensitivity", type=int, default=10,
                        help="Detection sensitivity (1-10, where 1 is strict and 10 is permissive, default: 5)")

    args = parser.parse_args()
    
    if args.pdf_file:
        # Process single file
        input_path = args.pdf_file
        if not os.path.exists(input_path):
            # Try with the default path prefix
            input_path = os.path.join("String_match/input_pdf", args.pdf_file)
            if not os.path.exists(input_path):
                print(f"Error: File not found: {args.pdf_file} or {input_path}")
                sys.exit(1)
        
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_pdf = os.path.join("String_match/output", f"{base_name}_highlighted.pdf")
        
        os.makedirs("String_match/output", exist_ok=True)
        
        # Load target strings
        target_strings = load_target_strings(args.targets)
        
        # Create a log file for single-file processing
        log_path = os.path.join("String_match/output", f"{base_name}_log.txt")
        with open(log_path, 'w', encoding='utf-8') as log_file:
            log_file.write(f"Processing file: {input_path}\n")
            log_file.write(f"Target strings: {', '.join(target_strings)}\n\n")
            
            # Start timing
            start_time = time.time()
            
            # Process the file with sensitivity parameter
            occurrences = highlight_strings_in_pdf(
                input_path, output_pdf, target_strings, 
                max_workers=args.cores, dpi=args.dpi, sensitivity=args.sensitivity
            )
            
            # End timing
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Log results
            log_file.write("Occurrences detected:\n")
            for target, count in occurrences.items():
                log_file.write(f"- '{target}': {count}\n")
            
            total_count = sum(occurrences.values())
            log_file.write(f"\nTotal occurrences: {total_count}\n")
            log_file.write(f"Processing time: {processing_time:.2f} seconds\n")
            
            print(f"Processing complete: {total_count} occurrences found in {processing_time:.2f} seconds")
            print(f"Results saved to {output_pdf}")
            print(f"Log saved to {log_path}")
    else:
        # Process all PDFs in the folder with sensitivity parameter
        process_all_pdfs(target_file=args.targets, max_workers=args.cores, dpi=args.dpi, sensitivity=args.sensitivity)