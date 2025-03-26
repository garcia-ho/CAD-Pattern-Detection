import cv2
import numpy as np
import os
from pathlib import Path
import pdf2image
from tqdm import tqdm
import concurrent.futures
from functools import partial
from PIL import Image, ImageChops
import tempfile

# Increase PIL image size limit - we'll handle large images properly
Image.MAX_IMAGE_PIXELS = None

class CADPreprocessor:
    def __init__(self, dpi=400):
        """
        Initialize the CAD drawing preprocessor.
        
        Args:
            dpi: Resolution for PDF conversion (dots per inch)
        """
        self.dpi = dpi
        self.tile_size = 4000  # Size of tiles for processing large images
        
    def preprocess_image(self, image):
        """
        Preprocess image to optimize for black pattern detection.
        Returns a binary image but in 3-channel BGR format for compatibility with annotation.
        """
        # If already grayscale, convert to BGR first
        if len(image.shape) < 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Create a white BGR canvas
        processed_image = np.ones_like(image) * 255
        
        # Create binary mask for dark pixels (adapt this threshold as needed)
        if len(image.shape) == 3:
            dark_mask = np.logical_and.reduce((
                image[:,:,0] < 128,
                image[:,:,1] < 128,
                image[:,:,2] < 128
            ))
        
        # Set dark pixels to black (but maintain 3 channels)
        processed_image[dark_mask] = [0, 0, 0]  # [B,G,R] = Black
        
        return processed_image  # Return BGR 3-channel image
    
    def preprocess_large_image_tiles(self, pil_image):
        """
        Process a large image by splitting it into tiles and processing each tile.
        
        Args:
            pil_image: PIL Image object
            
        Returns:
            Processed PIL Image (RGB mode)
        """
        width, height = pil_image.size
        print(f"Processing large image of size {width}x{height} using tiling")
        
        # Convert to RGB mode if needed
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
            
        # Create output image in RGB mode (PIL uses RGB)
        output_image = Image.new('RGB', (width, height), color=(255, 255, 255))
        
        # Calculate number of tiles
        n_cols = (width + self.tile_size - 1) // self.tile_size
        n_rows = (height + self.tile_size - 1) // self.tile_size
        
        total_tiles = n_rows * n_cols
        print(f"Splitting into {total_tiles} tiles ({n_rows}x{n_cols})")
        
        # Process each tile
        with tqdm(total=total_tiles, desc="Processing tiles") as pbar:
            for row in range(n_rows):
                for col in range(n_cols):
                    # Calculate tile boundaries
                    left = col * self.tile_size
                    upper = row * self.tile_size
                    right = min(left + self.tile_size, width)
                    lower = min(upper + self.tile_size, height)
                    
                    # Extract tile
                    tile = pil_image.crop((left, upper, right, lower))
                    
                    # Convert to numpy array for OpenCV (RGB to BGR)
                    tile_np = cv2.cvtColor(np.array(tile), cv2.COLOR_RGB2BGR)
                    
                    # Process the tile
                    processed_tile = self.preprocess_image(tile_np)
                    
                    # Convert back to PIL (BGR to RGB)
                    processed_rgb = cv2.cvtColor(processed_tile, cv2.COLOR_BGR2RGB)
                    processed_pil = Image.fromarray(processed_rgb)
                    
                    output_image.paste(processed_pil, (left, upper))
                    
                    pbar.update(1)
        
        return output_image  # RGB image (PIL format) with binary content

    def pdf_to_images(self, pdf_path, output_dir):
        """
        Convert a PDF file to high-resolution PNG images.
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save PNG images
            
        Returns:
            List of paths to generated PNG images
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert Path objects to string for compatibility
        pdf_path_str = str(pdf_path)
        
        # Get base name of the PDF without extension
        base_name = os.path.splitext(os.path.basename(pdf_path_str))[0]
        
        try:
            # Convert PDF to images with specified DPI
            print(f"Converting {pdf_path_str} to PNG at {self.dpi} DPI...")
            
            try:
                # Use poppler's pdf2image for conversion
                images = pdf2image.convert_from_path(
                    pdf_path_str, 
                    dpi=self.dpi,
                    fmt="png"
                )
            except Exception as e:
                print(f"Error in standard PDF conversion: {e}")
                print("Trying alternative PDF conversion approach...")
                
                # Use a more memory-efficient approach for very large PDFs
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_pattern = os.path.join(temp_dir, "page_%d.png")
                    # Convert directly to files with a temp pattern
                    pdf2image.convert_from_path(
                        pdf_path_str,
                        dpi=self.dpi,
                        output_folder=temp_dir,
                        output_file="page",
                        fmt="png"
                    )
                    # Load the images sequentially
                    images = []
                    for i, temp_file in enumerate(sorted(os.listdir(temp_dir))):
                        if temp_file.startswith("page_") and temp_file.endswith(".png"):
                            img_path = os.path.join(temp_dir, temp_file)
                            with Image.open(img_path) as img:
                                # Make a copy to keep after the tempfile is deleted
                                images.append(img.copy())
            
            # Save each page as an individual PNG
            png_paths = []
            for i, image in enumerate(images):
                # For multi-page PDFs, append page number
                if len(images) > 1:
                    output_path = os.path.join(output_dir, f"{base_name}_page{i+1}.png")
                else:
                    output_path = os.path.join(output_dir, f"{base_name}.png")
                
                # Check image size
                width, height = image.size
                pixels = width * height
                
                # For very large images, use tile-based processing
                if pixels > 50000000:  # ~50 megapixels
                    print(f"Large image detected ({width}x{height}), using tile-based processing")
                    processed_image = self.preprocess_large_image_tiles(image)
                    processed_image.save(output_path)
                else:
                    # Save the unprocessed image first
                    image.save(output_path)
                
                png_paths.append(output_path)
                
            print(f"Converted {pdf_path_str} to {len(png_paths)} PNG images")
            return png_paths
            
        except Exception as e:
            print(f"Error converting {pdf_path_str}: {e}")
            return []

    def preprocess_file(self, input_path, output_dir):
        """
        Preprocess a single file (PDF or image).
        
        Args:
            input_path: Path to input file
            output_dir: Directory to save processed image
            
        Returns:
            Path to processed image or None if failed
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert Path objects to string for compatibility
        input_path_str = str(input_path)
        
        try:
            # Get file extension
            ext = os.path.splitext(input_path_str)[1].lower()
            
            if ext == '.pdf':
                # Convert PDF to PNG images
                png_paths = self.pdf_to_images(input_path, output_dir)
                
                # Process each PNG image
                processed_paths = []
                for png_path in png_paths:
                    try:
                        # Check if the image is very large
                        with Image.open(png_path) as pil_img:
                            width, height = pil_img.size
                            pixels = width * height
                            
                        if pixels > 50000000:  # ~50 megapixels
                            # Large image already processed during PDF conversion
                            processed_paths.append(png_path)
                            continue
                        
                        # For smaller images, use standard processing
                        image = cv2.imread(png_path)
                        
                        if image is None:
                            print(f"Error: Could not load image {png_path}")
                            continue
                        
                        # Preprocess the image
                        processed = self.preprocess_image(image)
                        
                        # Save the processed image (overwrite the PNG)
                        cv2.imwrite(png_path, processed)
                        processed_paths.append(png_path)
                        
                    except Exception as e:
                        print(f"Error processing PNG {png_path}: {e}")
                    
                print(f"Processed {len(processed_paths)} images from {input_path_str}")
                return processed_paths
            else:
                # For non-PDF files, check if it's a large image
                try:
                    with Image.open(input_path_str) as pil_img:
                        width, height = pil_img.size
                        pixels = width * height
                        
                        # Get base name of the file without extension
                        base_name = os.path.splitext(os.path.basename(input_path_str))[0]
                        output_path = os.path.join(output_dir, f"{base_name}.png")
                        
                        if pixels > 50000000:  # ~50 megapixels
                            # Process large image using tiles
                            processed_image = self.preprocess_large_image_tiles(pil_img)
                            processed_image.save(output_path)
                            return [output_path]
                        
                        # For regular sized images, use OpenCV
                        image = cv2.imread(input_path_str)
                        
                        if image is None:
                            print(f"Error: Could not load image {input_path_str}")
                            return None
                        
                        # Preprocess the image
                        processed = self.preprocess_image(image)
                        
                        # Save the processed image
                        cv2.imwrite(output_path, processed)
                        print(f"Processed {input_path_str} -> {output_path}")
                        
                        return [output_path]
                        
                except Exception as e:
                    print(f"Error with PIL, trying OpenCV directly: {e}")
                    
                    # Fall back to OpenCV
                    image = cv2.imread(input_path_str)
                    
                    if image is None:
                        print(f"Error: Could not load image {input_path_str}")
                        return None
                    
                    # Preprocess the image
                    processed = self.preprocess_image(image)
                    
                    # Get base name of the file without extension
                    base_name = os.path.splitext(os.path.basename(input_path_str))[0]
                    output_path = os.path.join(output_dir, f"{base_name}.png")
                    
                    # Save the processed image
                    cv2.imwrite(output_path, processed)
                    print(f"Processed {input_path_str} -> {output_path}")
                    
                    return [output_path]
                
        except Exception as e:
            print(f"Error processing {input_path_str}: {e}")
            return None

def preprocess_directory(input_dir, output_dir, dpi=400, workers=4):
    """
    Process all files in a directory.
    
    Args:
        input_dir: Input directory containing PDF/image files
        output_dir: Directory to save processed images
        dpi: Resolution for PDF conversion (dots per inch)
        workers: Number of worker threads for parallel processing
        
    Returns:
        List of all processed image paths
    """
    # Create preprocessor
    preprocessor = CADPreprocessor(dpi=dpi)
    
    # Find all files in input directory
    input_files = []
    
    # Look for PDF files first
    pdf_files = list(Path(input_dir).glob('*.pdf'))
    input_files.extend(pdf_files)
    
    # Then look for image files
    image_exts = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']
    for ext in image_exts:
        image_files = list(Path(input_dir).glob(ext))
        input_files.extend(image_files)
    
    print(f"Found {len(input_files)} files in {input_dir}")
    
    # Process each file
    all_processed = []
    
    # For safety with large images, use parallel processing only for multiple files
    # but process each file's tiles sequentially to avoid memory issues
    if workers > 1 and len(input_files) > 1:
        print(f"Processing files in parallel with {workers} workers...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            preprocess_func = partial(preprocessor.preprocess_file, output_dir=output_dir)
            
            # Process files with progress bar
            results = list(tqdm(
                executor.map(preprocess_func, input_files), 
                total=len(input_files),
                desc="Processing files"
            ))
            
            # Flatten results list (since each file may produce multiple images)
            for result in results:
                if result:
                    all_processed.extend(result)
    else:
        # Process sequentially
        print("Processing files sequentially...")
        for input_file in tqdm(input_files, desc="Processing files"):
            result = preprocessor.preprocess_file(input_file, output_dir)
            if result:
                all_processed.extend(result)
    
    print(f"Processed {len(all_processed)} images from {len(input_files)} input files")
    return all_processed