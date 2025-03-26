#!/usr/bin/env python3
# filepath: /home/r27user6/CAD_matching/String_match/preprocess_text.py

import os
import argparse
import glob
import time
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image


def convert_pdf_to_image(pdf_path, output_path, dpi=400):
    """
    Convert a single-page PDF to a high-resolution image
    
    Args:
        pdf_path (str): Path to the PDF file
        output_path (str): Path to save the output image
        dpi (int): DPI resolution for the output image
    
    Returns:
        bool: Success status
    """
    try:
        # Open the PDF
        doc = fitz.open(pdf_path)
        
        # Ensure we only process the first page
        if len(doc) > 0:
            # Get the first page
            page = doc[0]
            
            # Render page to image
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72), alpha=False)
            
            # Convert to numpy array
            if pix.n == 1:  # Grayscale
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
                img_rgb = np.stack([img, img, img], axis=2)
            elif pix.n == 3:  # RGB
                img_rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            elif pix.n == 4:  # RGBA
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 4)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            
            # Save image
            cv2.imwrite(output_path, img_rgb)
            
            doc.close()
            return True
        
        doc.close()
        return False
        
    except Exception as e:
        print(f"Error converting PDF to image: {e}")
        return False


def preprocess_image(image_path, resize_factor=1.0):
    """
    Load an image and resize it if needed
    
    Args:
        image_path (str): Path to the image file
        resize_factor (float): Factor by which to resize the image
    
    Returns:
        PIL.Image: Loaded and resized image
    """
    # Load image
    if isinstance(image_path, str):
        img = Image.open(image_path)
    else:
        img = image_path
        
    # Resize if needed
    if resize_factor != 1.0:
        new_width = int(img.width * resize_factor)
        new_height = int(img.height * resize_factor)
        img = img.resize((new_width, new_height), Image.LANCZOS)
        
    return img


def remove_noise(image):
    """
    Remove noise from an image
    
    Args:
        image (PIL.Image): Input image
    
    Returns:
        PIL.Image: Denoised image
    """
    # Convert to numpy array
    img_np = np.array(image)
    
    # Convert to grayscale if needed
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np
    
    # Apply bilateral filter
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply non-local means denoising
    denoised = cv2.fastNlMeansDenoising(bilateral, h=10)
    
    # Apply median blur
    median = cv2.medianBlur(denoised, 3)
    
    # Return as PIL image
    return Image.fromarray(median)


def enhance_text(image):
    """
    Enhance text in an image
    
    Args:
        image (PIL.Image): Input image
    
    Returns:
        PIL.Image: Enhanced image
    """
    # Convert to numpy array
    img_np = np.array(image)
    
    # Convert to grayscale if needed
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np
    
    # Create a CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Return as PIL image
    return Image.fromarray(enhanced)


def adjust_contrast(image):
    """
    Adjust contrast using CLAHE
    
    Args:
        image (PIL.Image): Input image
    
    Returns:
        PIL.Image: Contrast-adjusted image
    """
    # Convert to numpy array
    img_np = np.array(image)
    
    # Convert to grayscale if needed
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Return as PIL image
    return Image.fromarray(enhanced)


def sharpen_image(image):
    """
    Sharpen an image using unsharp masking
    
    Args:
        image (PIL.Image): Input image
    
    Returns:
        PIL.Image: Sharpened image
    """
    # Convert to numpy array
    img_np = np.array(image)
    
    # Convert to grayscale if needed
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np
    
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    
    # Apply unsharp masking
    sharpened = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    
    # Return as PIL image
    return Image.fromarray(sharpened)


def binarize_image(image):
    """
    Convert image to binary using adaptive thresholding
    
    Args:
        image (PIL.Image): Input image
    
    Returns:
        PIL.Image: Binarized image
    """
    # Convert to numpy array
    img_np = np.array(image)
    
    # Convert to grayscale if needed
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Return as PIL image
    return Image.fromarray(binary)


def remove_lines(image):
    """
    Remove horizontal and vertical lines from an image
    
    Args:
        image (PIL.Image): Input image
    
    Returns:
        PIL.Image: Image with lines removed
    """
    # Convert to numpy array
    img_np = np.array(image)
    
    # Convert to grayscale if needed
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np
    
    # Create a binary image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    # Detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    
    # Combine horizontal and vertical lines
    lines = cv2.add(horizontal_lines, vertical_lines)
    
    # Remove lines from original image
    result = gray.copy()
    result[lines > 0] = 255
    
    # Return as PIL image
    return Image.fromarray(result)


def filter_by_color(image, threshold=128):
    """
    Filter image by color - keep only black text on white background
    
    Args:
        image (PIL.Image): Input image
        threshold (int): RGB threshold for filtering
    
    Returns:
        PIL.Image: Filtered image
    """
    # Convert to numpy array
    img_np = np.array(image)
    
    # Create a mask for black text
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        # RGB image
        black_mask = np.logical_and.reduce((
            img_np[:,:,0] < threshold,
            img_np[:,:,1] < threshold,
            img_np[:,:,2] < threshold
        ))
        
        # Create filtered image with white background
        filtered = np.ones_like(img_np) * 255
        filtered[black_mask] = [0, 0, 0]
    else:
        # Grayscale image
        black_mask = img_np < threshold
        filtered = np.ones_like(img_np) * 255
        filtered[black_mask] = 0
    
    # Return as PIL image
    return Image.fromarray(filtered)


def advanced_preprocess(image_path, output_path):
    """
    Apply comprehensive preprocessing to enhance text visibility
    Args:
        image_path (str): Path to input image
        output_path (str): Path to save processed image
    
    Returns:
        bool: Success status
    """
    try:
        # Load the image with OpenCV directly
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"Error: Could not load image {image_path}")
            return False
        
        # STEP 1: FILTER GREY - Remove all pixels with RGB values >= 128,128,128
        # Create a mask where all RGB values must be < 128 (non-grey elements)
        black_mask = np.logical_and.reduce((
            img[:,:,0] < 128,
            img[:,:,1] < 128,
            img[:,:,2] < 128
        ))

        # Create a new image with white background
        filtered_img = np.ones_like(img) * 255  # White background

        # Set the black text pixels
        filtered_img[black_mask] = [0, 0, 0]  # Black text

        # STEP 2: CONVERT TO GRAYSCALE
        gray = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY)

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
        
        # Save the processed image
        cv2.imwrite(output_path, enhanced)
        
        return True
        
    except Exception as e:
        print(f"Error during advanced preprocessing: {e}")
        return False


def process_pdfs(input_dir="String_match/input_pdf", output_dir="String_match/filtered_img", 
                dpi=400, advanced=True):
    """
    Process single-page PDF files - convert to images and preprocess
    
    Args:
        input_dir (str): Directory containing input PDF files
        output_dir (str): Directory to save processed images
        dpi (int): DPI for PDF to image conversion
        advanced (bool): Use advanced preprocessing
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all PDF files in input directory
    pdf_files = glob.glob(os.path.join(input_dir, "*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Process each PDF file
    for pdf_idx, pdf_path in enumerate(pdf_files, 1):
        pdf_basename = os.path.basename(pdf_path)
        pdf_name = os.path.splitext(pdf_basename)[0]
        
        print(f"Processing PDF {pdf_idx}/{len(pdf_files)}: {pdf_basename}")
        
        start_time = time.time()
        
        try:
            # Define temporary and final output paths
            temp_image_path = os.path.join(output_dir, f"temp_{pdf_name}.jpg")
            final_image_path = os.path.join(output_dir, f"processed_{pdf_name}.jpg")
            
            # Step 1: Convert PDF to image
            success = convert_pdf_to_image(pdf_path, temp_image_path, dpi=dpi)
            
            if not success:
                print(f"Failed to convert {pdf_basename} to image")
                continue
                
            # Step 2: Preprocess the image
            try:
                if advanced:
                    # Use advanced preprocessing pipeline
                    success = advanced_preprocess(temp_image_path, final_image_path)
                    if success:
                        print(f"Advanced preprocessing completed for {pdf_basename}")
                    else:
                        print(f"Advanced preprocessing failed for {pdf_basename}")
                else:
                    # Use simple preprocessing pipeline
                    img = preprocess_image(temp_image_path)
                    img = filter_by_color(img)
                    img = remove_noise(img)
                    img = remove_lines(img)
                    img = enhance_text(img)
                    img = sharpen_image(img)
                    
                    # Save the processed image
                    img.save(final_image_path)
                    print(f"Basic preprocessing completed for {pdf_basename}")
                    
                # Remove the temporary image
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
                    
            except Exception as e:
                print(f"Error processing image for {pdf_basename}: {str(e)}")
                continue
            
            # Log completion time
            processing_time = time.time() - start_time
            print(f"Completed processing {pdf_basename} in {processing_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error processing PDF {pdf_basename}: {str(e)}")
            continue
    
    print("All PDFs processed successfully")
