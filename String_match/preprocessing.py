# filepath: /home/r27user6/CAD_matching/String_match/preprocessing.py

import os
import re
import fitz  # PyMuPDF for PDF handling
import cv2
import numpy as np
from PIL import Image


def process_pdf_page_with_paddle(args):
    """
    Process a single PDF page using PaddleOCR for better text detection.
    
    Args:
        args (tuple): (doc_path, page_num, target_strings, dpi, sensitivity)
    
    Returns:
        dict: Contains page number, found boxes, and processing information
    """
    from paddleocr import PaddleOCR
    
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
            
            # Special handling for VCD - add common misrecognitions 
            if target == 'VCD':
                variants.extend(['VC0', 'V CD', 'V-CD', 'VCO', 'VG0', 'VC D'])
                # Add lowercase variants for VCD which is often misread
                variants.extend(['vcd', 'vco', 'vcd', 'ved'])
                
            # Store all unique variants
            target_variants[i] = list(set(variants))
        
        # ENHANCED PREPROCESSING: FILTER GREY, REMOVE LINES, APPLY NOISE REDUCTION
        print(f"Processing PDF: Filtering grey elements and cleaning with noise reduction...")

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
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        horizontal_lines = cv2.morphologyEx(255 - gray, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

        # Detect and remove LONG vertical lines only
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
        vertical_lines = cv2.morphologyEx(255 - gray, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

        # Combine horizontal and vertical lines
        lines = cv2.add(horizontal_lines, vertical_lines)

        # Remove lines from the image (set to white)
        img_no_lines[lines > 0] = 255

        # STEP 4: APPLY NOISE REDUCTION TECHNIQUES
        bilateral_filtered = cv2.bilateralFilter(img_no_lines, d=5, sigmaColor=75, sigmaSpace=75)
        denoised = cv2.fastNlMeansDenoising(bilateral_filtered, h=10, templateWindowSize=7, searchWindowSize=21)
        median_filtered = cv2.medianBlur(denoised, 3)

        # STEP 5: CLEAN UP THE IMAGE
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(median_filtered, cv2.MORPH_OPEN, kernel, iterations=1)

        # STEP 6: ENHANCE CONTRAST FOR BETTER TEXT READABILITY
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(cleaned)

        # Create output directory
        os.makedirs('String_match/filtered_img', exist_ok=True)

        # Save processed image for OCR
        temp_img_path = f"String_match/filtered_img/{base_filename}_temp.jpg"
        cv2.imwrite(temp_img_path, enhanced)

        # INITIALIZE PADDLE OCR
        # Use different models based on the content type
        ocr_engine = PaddleOCR(
            use_angle_cls=True,  # Enable rotation detection
            lang="en",           # English language
            show_log=False,      # Don't show log
            use_gpu=False        # Can set to True if GPU available
        )
        
        # Run PaddleOCR detection and recognition
        print("Running PaddleOCR detection...")
        paddle_results = ocr_engine.ocr(temp_img_path, cls=True)
        
        if paddle_results is not None and len(paddle_results) > 0:
            # Process OCR results
            for line in paddle_results[0]:  # First page results
                # Each line has [[points], (text, confidence)]
                box_points = line[0]  # 4 points of the detection box
                text, confidence = line[1]  # Text and confidence
                
                # Skip low confidence detections
                min_confidence = max(0.5, 0.7 - (sensitivity * 0.05))
                
                # Lower threshold for VCD
                if "VCD" in text.upper() or "VC D" in text.upper():
                    min_confidence = max(0.3, 0.5 - (sensitivity * 0.05))
                    
                if confidence < min_confidence:
                    continue
                
                # Calculate bounding box coordinates
                x_coords = [point[0] for point in box_points]
                y_coords = [point[1] for point in box_points]
                
                # Get bounding box in x, y, w, h format
                x = min(x_coords)
                y = min(y_coords)
                w = max(x_coords) - x
                h = max(y_coords) - y
                
                # Skip extremely small or large boxes
                if w < 5 or h < 5:
                    continue
                    
                if w > enhanced.shape[1] * 0.9 or h > enhanced.shape[0] * 0.9:
                    continue
                
                # Scale coordinates back to original PDF size
                scale_to_pdf = 72 / dpi
                pdf_x = x * scale_to_pdf
                pdf_y = y * scale_to_pdf
                pdf_w = w * scale_to_pdf
                pdf_h = h * scale_to_pdf
                
                # Normalize detected text
                normalized_text = text.strip().upper()
                cleaned_text = re.sub(r'[^\w\s\-\.]', '', normalized_text)
                
                # Check against target strings
                for target_idx, target_upper in enumerate(target_strings_upper):
                    target_string = target_strings[target_idx]
                    curr_variants = target_variants.get(target_idx, [target_upper])
                    
                    for variant in curr_variants:
                        found_match = False
                        match_type = ""
                        
                        # Check different match types
                        if variant == normalized_text or variant == cleaned_text:
                            found_match = True
                            match_type = "exact"
                        
                        elif variant in normalized_text.split() or variant in cleaned_text.split():
                            found_match = True
                            match_type = "word"
                        
                        elif variant in normalized_text or variant in cleaned_text:
                            found_match = True
                            match_type = "contained"
                        
                        # Special case for VCD
                        if target_string.upper() == 'VCD' and not found_match:
                            if (normalized_text.startswith('V') and 
                                ('C' in normalized_text or '0' in normalized_text or 'O' in normalized_text) and
                                ('D' in normalized_text or 'O' in normalized_text or '0' in normalized_text)):
                                found_match = True
                                match_type = "vcd_special"
                        
                        # If found a match, add to results
                        if found_match:
                            all_boxes.append({
                                'x': pdf_x, 'y': pdf_y, 'w': pdf_w, 'h': pdf_h,
                                'text': text, 'conf': float(confidence * 100),
                                'target': target_string,
                                'source': "paddleocr",
                                'match_type': match_type,
                                'variant': variant
                            })
                            break
        
        # Clean up temporary image
        try:
            os.remove(temp_img_path)
        except:
            pass
        
        # SAVE PDF OUTPUT
        pdf_path = f"String_match/filtered_img/{base_filename}.pdf"
        try:
            # Convert enhanced image to JPEG with high compression first
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
            _, jpg_data = cv2.imencode('.jpg', enhanced, encode_param)
            
            # Create a temporary JPEG file
            temp_jpg_path = f"String_match/filtered_img/{base_filename}_temp.jpg"
            with open(temp_jpg_path, 'wb') as f:
                f.write(jpg_data)
            
            # Use pymupdf to create an optimized PDF
            output_pdf = fitz.open()
            
            # Create new page with downscaled dimensions if original is very large
            scale_factor = 1.0
            if page_width > 1000 or page_height > 1000:
                scale_factor = 0.75
            
            new_width = page_width * scale_factor
            new_height = page_height * scale_factor
            pdf_page = output_pdf.new_page(width=new_width, height=new_height)
            
            # Insert the JPEG image with maximum compression
            img_rect = fitz.Rect(0, 0, new_width, new_height)
            pdf_page.insert_image(img_rect, filename=temp_jpg_path, keep_proportion=True)
            
            # Use maximum compression settings when saving
            output_pdf.save(pdf_path, 
                          garbage=4,
                          deflate=True,
                          clean=True,
                          linear=True)
            
            output_pdf.close()
            
            # Remove temporary file
            try:
                os.remove(temp_jpg_path)
            except:
                pass
            
            print(f"Saved filtered PDF to {pdf_path}")
            
        except Exception as e:
            print(f"Error saving PDF with PyMuPDF: {e}")
            
            # Fallback method: Save as JPEG with high compression
            fallback_jpg_path = f"String_match/filtered_img/{base_filename}_filtered.jpg" 
            cv2.imwrite(fallback_jpg_path, enhanced, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            print(f"Saved fallback JPEG to {fallback_jpg_path}")
        
        # Close the document
        doc.close()
        
        print(f"Found {len(all_boxes)} potential matches before deduplication")
        
        # DEDUPLICATION (same as your current code)
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
                    "vcd_special": 3,  # High priority for special VCD detection
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
        
        print(f"Found {len(merged_boxes)} unique matches after deduplication")
        
        return {
            'page_num': page_num,
            'boxes': merged_boxes,
            'scale': 1.0
        }
        
    except Exception as e:
        print(f"Error processing page: {e}")
        import traceback
        traceback.print_exc()
        return {
            'page_num': page_num,
            'boxes': [],
            'error': str(e)
        }

if __name__ == "__main__":
    print("This module contains preprocessing functions for PDF files.")
    print("Please use main_text_det.py to run the full processing pipeline.")