import os
import logging
import numpy as np
from src.detector import CADObjectDetector

# Configure logging
logger = logging.getLogger(__name__)

def detect_objects(self, cad_path, target_path, output_path=None, method='hybrid', threshold=0.7):
        """
        Detect standard CAD valve symbols using pattern matching
        
        Args:
            cad_path (str): Path to the CAD PNG file
            target_path (str): Path to the target PNG image (standard valve symbol)
            output_path (str, optional): Path where to save the highlighted CAD image
            method (str): Matching method: 'pattern', 'geometric', or 'hybrid'
            threshold (float): Matching threshold (0.7-0.9 recommended)
        
        Returns:
            tuple: (Number of occurrences, Path to highlighted CAD image)
        """
        logger.info(f"Processing CAD file: {cad_path}")
        logger.info(f"Looking for pattern: {target_path}")
        logger.info(f"Using method: {method}")
        
        # Load images in grayscale - simplest and most reliable for pattern matching
        cad_gray = cv2.imread(cad_path, cv2.IMREAD_GRAYSCALE)
        target_gray = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        
        if cad_gray is None:
            raise ValueError(f"Could not load CAD image: {cad_path}")
        
        if target_gray is None:
            raise ValueError(f"Could not load target image: {target_path}")
            
        # Load color version for highlighting
        cad_color = cv2.imread(cad_path)
        
        logger.info(f"CAD image size: {cad_gray.shape[1]}x{cad_gray.shape[0]}")
        logger.info(f"Target image size: {target_gray.shape[1]}x{target_gray.shape[0]}")
        
        # Normalize intensity values
        cad_norm = cv2.normalize(cad_gray, None, 0, 255, cv2.NORM_MINMAX)
        target_norm = cv2.normalize(target_gray, None, 0, 255, cv2.NORM_MINMAX)
        
        # Choose matching method
        if method == 'pattern':
            # Use only template matching with multi-scale support
            matches, highlighted_img = self._multi_scale_template_matching(cad_norm, cad_color, target_norm, threshold)
        elif method == 'geometric':
            # Use only geometric feature matching
            matches, highlighted_img = self._multi_scale_geometric_matching(cad_norm, cad_color, target_norm, threshold)
        else:
            # Default: hybrid approach (combining both methods)
            matches, highlighted_img = self._multi_scale_hybrid_matching(cad_norm, cad_color, target_norm, threshold)
        
        match_count = len(matches)
        
        logger.info(f"Found {match_count} matches with confidence >= {threshold}")
        
        # Save highlighted image if specified and matches found
        highlighted_path = None
        if output_path and match_count > 0:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            cv2.imwrite(output_path, highlighted_img)
            highlighted_path = output_path
            logger.info(f"Saved highlighted image to: {output_path}")
        
        return match_count, highlighted_path