import os
import cv2
import numpy as np
import logging
import math
from pathlib import Path
from skimage.morphology import skeletonize
from skimage.measure import find_contours, approximate_polygon

# Configure logging
logger = logging.getLogger(__name__)

class CADObjectDetector:
    def __init__(self):
        pass
        
    def detect_objects(self, cad_path, target_path, output_path=None, method='pattern', threshold=0.7):
        """
        Detect standard CAD valve symbols using direct pattern matching
        
        Args:
            cad_path (str): Path to the CAD PNG file
            target_path (str): Path to the target PNG image (standard valve symbol)
            output_path (str, optional): Path where to save the highlighted CAD image
            method (str): Only 'pattern' is supported
            threshold (float): Matching threshold (0.7-0.9 recommended)
        
        Returns:
            tuple: (Number of occurrences, Path to highlighted CAD image)
        """
        logger.info(f"Processing CAD file: {cad_path}")
        logger.info(f"Looking for pattern: {target_path}")
        
        # Load images in grayscale - simplest and most reliable for exact pattern matching
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
        
        # For identical images, minimal preprocessing is needed
        # Just ensure we have normalized intensity values
        cad_norm = cv2.normalize(cad_gray, None, 0, 255, cv2.NORM_MINMAX)
        target_norm = cv2.normalize(target_gray, None, 0, 255, cv2.NORM_MINMAX)
        
        # Get direct pattern matches with a high confidence
        matches, highlighted_img = self._exact_pattern_matching(cad_norm, cad_color, target_norm, threshold)
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
    
    def _exact_pattern_matching(self, cad_img, cad_color, target_img, threshold):
        """
        Perform exact pattern matching with minimal processing
        """
        h, w = target_img.shape
        matches = []
        
        # For identical patterns, direct template matching should work perfectly
        result = cv2.matchTemplate(cad_img, target_img, cv2.TM_CCOEFF_NORMED)
        
        # Find all locations where the match score exceeds threshold
        locations = np.where(result >= threshold)
        
        # Convert to x,y coordinates
        for pt in zip(*locations[::-1]):
            matches.append((pt[0], pt[1], w, h))
        
        # If no matches found and this is unexpected, try with slightly lower threshold
        if len(matches) == 0 and threshold > 0.8:
            logger.debug("No matches found with high threshold, trying with threshold=0.75")
            locations = np.where(result >= 0.75)
            for pt in zip(*locations[::-1]):
                matches.append((pt[0], pt[1], w, h))
        
        # Filter overlapping detections
        matches = self._filter_overlaps(matches)
        
        # Draw red rectangles around matches
        highlighted_img = cad_color.copy()
        for (x, y, w, h) in matches:
            # Draw red rectangle (BGR format: 0,0,255)
            cv2.rectangle(highlighted_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            # Get match confidence for this location
            confidence = result[y, x]
            # Add confidence score text
            text = f"{confidence:.2f}"
            cv2.putText(highlighted_img, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 0, 255), 1, cv2.LINE_AA)
        
        return matches, highlighted_img
    
    def _filter_overlaps(self, matches, overlap_thresh=0.3):
        """
        Filter overlapping detections using Non-Maximum Suppression
        """
        if not matches:
            return []
        
        # Convert to format expected by NMSBoxes
        boxes = [[x, y, x+w, y+h] for (x, y, w, h) in matches]
        scores = np.ones(len(matches))  # Equal weights
        
        try:
            # Apply non-maximum suppression
            indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, 
                                     nms_threshold=overlap_thresh)
            
            # Extract the filtered matches
            if len(indices) > 0:
                if isinstance(indices, tuple) or (isinstance(indices, np.ndarray) and indices.ndim > 1):
                    indices = indices.flatten()
                filtered_matches = [matches[i] for i in indices]
                return filtered_matches
        except Exception as e:
            logger.warning(f"Error in NMS filtering: {e}")
            # Fallback: simply return all matches if NMS fails
        
        return matches

def preprocess_cad_graph(image_path):
    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Ensure binary image (black and white)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Skeletonize to get thin representation of lines
    skeleton = skeletonize(binary / 255).astype(np.uint8) * 255
    
    # Extract basic geometric features
    features = extract_geometric_features(skeleton)
    
    return skeleton, features

def extract_geometric_features(skeleton_img):
    # Find contours
    contours = find_contours(skeleton_img, 0.5)
    
    features = {
        'lines': [],
        'corners': [],
        'junctions': []
    }
    
    for contour in contours:
        # Approximate polygon to get line segments
        approx = approximate_polygon(contour, tolerance=2.0)
        
        # Extract lines
        for i in range(len(approx) - 1):
            p1, p2 = approx[i], approx[i + 1]
            line = (p1, p2)
            features['lines'].append(line)
            
        # Detect corners (significant angle changes)
        for i in range(1, len(approx) - 1):
            p1, p2, p3 = approx[i-1], approx[i], approx[i+1]
            v1 = p1 - p2
            v2 = p3 - p2
            angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            if angle < 2.5 and angle > 0.5:  # ~30 to ~150 degrees
                features['corners'].append(p2)
                
    # Find junctions (where 3+ lines meet) using Harris corners
    corners = cv2.cornerHarris(skeleton_img, 3, 3, 0.04)
    corners = cv2.dilate(corners, None)
    threshold = 0.01 * corners.max()
    junction_points = np.where(corners > threshold)
    for y, x in zip(junction_points[0], junction_points[1]):
        features['junctions'].append((x, y))
    
    return features

def match_cad_patterns(cad_features, target_features):
    # Pattern matching based on geometric features
    # This is a simplified matching function - you might need more sophisticated matching
    similarity_score = 0
    
    # Compare line counts and orientations
    line_count_diff = abs(len(cad_features['lines']) - len(target_features['lines']))
    line_score = 1 / (1 + line_count_diff)
    
    # Compare corner counts and positions
    corner_count_diff = abs(len(cad_features['corners']) - len(target_features['corners']))
    corner_score = 1 / (1 + corner_count_diff)
    
    # Compare junction counts
    junction_count_diff = abs(len(cad_features['junctions']) - len(target_features['junctions']))
    junction_score = 1 / (1 + junction_count_diff)
    
    # Weighted sum of scores
    similarity_score = 0.5 * line_score + 0.3 * corner_score + 0.2 * junction_score
    
    return similarity_score

