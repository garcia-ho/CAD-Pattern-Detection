import os
import cv2
import numpy as np
import fitz  # PyMuPDF
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CADObjectDetector:
    def __init__(self):
        self.methods = {
            'template': self._template_matching,
            'feature': self._feature_matching,
            'contour': self._contour_matching
        }
        # SIFT for feature extraction
        self.sift = cv2.SIFT_create(nfeatures=2000)  # Increased features for complex CADs
        # FLANN matcher parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=100)  # More checks for better accuracy
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        
    def detect_objects(self, cad_path, target_path, output_path=None, method='feature', threshold=0.7, dpi=300):
        """
        Detect objects in CAD file that match the target image
        
        Args:
            cad_path (str): Path to the CAD PDF file
            target_path (str): Path to the target PNG image
            output_path (str, optional): Path where to save the highlighted CAD file
            method (str): Detection method ('template', 'feature', 'contour')
            threshold (float): Matching threshold
            dpi (int): DPI for rendering PDF pages
        
        Returns:
            tuple: (Number of occurrences, Path to highlighted CAD file)
        """
        logger.info(f"Processing CAD file: {cad_path}")
        logger.info(f"Looking for target: {target_path}")
        
        # Load target image
        target_img = cv2.imread(target_path, cv2.IMREAD_COLOR)
        if target_img is None:
            raise ValueError(f"Could not load target image from {target_path}")
        target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        
        # Load CAD file with PyMuPDF
        doc = fitz.open(cad_path)
        pages = len(doc)
        logger.info(f"CAD file has {pages} page(s)")
        
        total_matches = 0
        all_matches = []
        
        # Calculate zoom factor based on DPI
        zoom = dpi / 72  # 72 is the base DPI for PDF
        matrix = fitz.Matrix(zoom, zoom)
        
        # Process each page in the PDF
        for page_num in range(pages):
            logger.info(f"Processing page {page_num+1}/{pages}")
            page = doc[page_num]
            
            # Get page original dimensions
            orig_width, orig_height = page.rect.width, page.rect.height
            
            # Convert PDF page to high-resolution image
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img_width, img_height = pix.width, pix.height
            
            # Calculate scale factor between rendered image and PDF coordinates
            scale_x = img_width / orig_width
            scale_y = img_height / orig_height
            
            # Convert to PIL image and then to numpy array for OpenCV
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            cad_img = np.array(img)
            cad_gray = cv2.cvtColor(cad_img, cv2.COLOR_RGB2GRAY)
            
            logger.info(f"Rendered page at {img_width}x{img_height} pixels")
            
            # Choose detection method
            if method in self.methods:
                matches, matched_img = self.methods[method](cad_gray, cad_img, target_gray, target_img, threshold)
                if matches:
                    # Rescale matches to original PDF coordinates
                    pdf_matches = []
                    for x, y, w, h in matches:
                        pdf_x = x / scale_x
                        pdf_y = y / scale_y
                        pdf_w = w / scale_x
                        pdf_h = h / scale_y
                        pdf_matches.append((pdf_x, pdf_y, pdf_w, pdf_h))
                    
                    # Store matches for this page
                    all_matches.append((page_num, pdf_matches, matched_img))
                    total_matches += len(matches)
                    logger.info(f"Found {len(matches)} matches on page {page_num+1}")
            else:
                raise ValueError(f"Method {method} not supported")
        
        # Highlight matches in PDF if output_path provided
        highlighted_path = None
        if output_path and all_matches:
            logger.info(f"Creating highlighted PDF at {output_path}")
            highlighted_path = self._highlight_in_pdf(doc, all_matches, output_path)
        else:
            # Close document
            doc.close()
        
        logger.info(f"Total matches found: {total_matches}")
        return total_matches, highlighted_path
    
    def _template_matching(self, cad_gray, cad_img, target_gray, target_img, threshold):
        """Template matching detection method"""
        # Try different scales if necessary
        scales = [1.0, 0.8, 1.2]
        matches = []
        best_matched_img = cad_img.copy()
        
        for scale in scales:
            logger.debug(f"Trying template matching with scale {scale}")
            if scale != 1.0:
                width = int(target_gray.shape[1] * scale)
                height = int(target_gray.shape[0] * scale)
                resized_target = cv2.resize(target_gray, (width, height))
            else:
                resized_target = target_gray
            
            h, w = resized_target.shape
            
            # Use multiple template matching methods for better results
            methods = [cv2.TM_CCOEFF_NORMED]
            
            for method in methods:
                result = cv2.matchTemplate(cad_gray, resized_target, method)
                locations = np.where(result >= threshold)
                
                scale_matches = []
                for y, x in zip(*locations[::-1]):
                    scale_matches.append((x, y, w, h))
                
                if scale_matches:
                    matches.extend(scale_matches)
        
        # Filter overlapping matches
        if matches:
            matches = self._filter_overlapping_matches(matches)
            
            # Draw the matches
            matched_img = cad_img.copy()
            for x, y, w, h in matches:
                cv2.rectangle(matched_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
            return matches, matched_img
        
        return [], best_matched_img
    
    def _feature_matching(self, cad_gray, cad_img, target_gray, target_img, threshold):
        """SIFT feature matching detection method"""
        logger.debug("Using SIFT feature matching")
        
        # Enhance image contrast for better feature detection
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        target_enhanced = clahe.apply(target_gray)
        cad_enhanced = clahe.apply(cad_gray)
        
        # Compute SIFT keypoints and descriptors
        kp_target, des_target = self.sift.detectAndCompute(target_enhanced, None)
        kp_cad, des_cad = self.sift.detectAndCompute(cad_enhanced, None)
        
        if des_target is None or des_cad is None or len(des_target) == 0 or len(des_cad) == 0:
            logger.warning("No keypoints found in one of the images")
            return [], cad_img.copy()
        
        logger.debug(f"Found {len(kp_target)} keypoints in target and {len(kp_cad)} in CAD")
        
        # FLANN based matching
        matches = self.flann.knnMatch(des_target, des_cad, k=2)
        
        # Filter matches using Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < threshold * n.distance:
                good_matches.append(m)
        
        logger.debug(f"Found {len(good_matches)} good feature matches")
        
        if len(good_matches) >= 4:
            # Get corresponding points
            src_pts = np.float32([kp_target[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_cad[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Use RANSAC to find homography
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
            
            # Get dimensions of target image
            h, w = target_gray.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            
            # Transform target image corners to find occurrences in CAD
            detected_instances = []
            
            # We need to cluster the good matches to find multiple instances
            clustered_points = self._cluster_matches(dst_pts)
            
            matched_img = cad_img.copy()
            for cluster in clustered_points:
                if len(cluster) >= 4:  # Need at least 4 points for homography
                    # Extract src and dst points for this cluster
                    cluster_indices = []
                    for i, dst_pt in enumerate(dst_pts):
                        for cluster_pt in cluster:
                            if np.array_equal(dst_pt, cluster_pt):
                                cluster_indices.append(i)
                                break
                    
                    if len(cluster_indices) >= 4:
                        cluster_src_pts = np.float32([kp_target[good_matches[i].queryIdx].pt for i in cluster_indices]).reshape(-1, 1, 2)
                        cluster_dst_pts = np.float32([kp_cad[good_matches[i].trainIdx].pt for i in cluster_indices]).reshape(-1, 1, 2)
                        
                        H, _ = cv2.findHomography(cluster_src_pts, cluster_dst_pts, cv2.RANSAC, 5.0)
                        if H is not None:
                            try:
                                dst = cv2.perspectiveTransform(pts, H)
                                # Convert to list of rectangles format (x, y, w, h)
                                x = min(pt[0][0] for pt in dst)
                                y = min(pt[0][1] for pt in dst)
                                max_x = max(pt[0][0] for pt in dst)
                                max_y = max(pt[0][1] for pt in dst)
                                detected_instances.append((int(x), int(y), int(max_x - x), int(max_y - y)))
                                
                                # Draw bounding box
                                cv2.polylines(matched_img, [np.int32(dst)], True, (0, 255, 0), 3)
                            except cv2.error as e:
                                logger.warning(f"Error during perspective transform: {e}")
            
            # Filter overlapping instances
            detected_instances = self._filter_overlapping_matches(detected_instances)
            
            return detected_instances, matched_img
        
        return [], cad_img.copy()
    
    def _contour_matching(self, cad_gray, cad_img, target_gray, target_img, threshold):
        """Contour matching detection method"""
        logger.debug("Using contour matching")
        
        # Apply binary thresholding to both images with multiple threshold levels
        matches_all = []
        thresholds = [127, 150, 180]
        
        for thresh in thresholds:
            # Apply threshold
            _, target_binary = cv2.threshold(target_gray, thresh, 255, cv2.THRESH_BINARY)
            _, cad_binary = cv2.threshold(cad_gray, thresh, 255, cv2.THRESH_BINARY)
            
            # Find contours
            target_contours, _ = cv2.findContours(target_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cad_contours, _ = cv2.findContours(cad_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            # Get the main contours from the target image
            if not target_contours:
                continue
            
            # Filter small contours
            min_area = 100
            target_contours = [c for c in target_contours if cv2.contourArea(c) > min_area]
            cad_contours = [c for c in cad_contours if cv2.contourArea(c) > min_area]
            
            if not target_contours:
                continue
                
            # Use multiple target contours for better matching
            for target_contour in sorted(target_contours, key=cv2.contourArea, reverse=True)[:3]:
                target_area = cv2.contourArea(target_contour)
                
                # Match contours
                for cad_contour in cad_contours:
                    cad_area = cv2.contourArea(cad_contour)
                    
                    # Filter by area ratio
                    area_ratio = cad_area / target_area if target_area > 0 else 0
                    if 0.5 <= area_ratio <= 2.0:
                        # Calculate shape match
                        match_score = cv2.matchShapes(target_contour, cad_contour, cv2.CONTOURS_MATCH_I2, 0)
                        if match_score < threshold:
                            x, y, w, h = cv2.boundingRect(cad_contour)
                            matches_all.append((x, y, w, h, match_score))
        
        # Sort matches by score and remove duplicates
        matches_all.sort(key=lambda m: m[4])
        
        # Extract bounding boxes
        matches = [(x, y, w, h) for x, y, w, h, _ in matches_all]
        
        # Filter overlapping matches
        matches = self._filter_overlapping_matches(matches)
        
        # Draw matches
        matched_img = cad_img.copy()
        for x, y, w, h in matches:
            cv2.rectangle(matched_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        return matches, matched_img
    
    def _filter_overlapping_matches(self, matches, overlap_threshold=0.5):
        """Filter overlapping bounding boxes using non-maximum suppression"""
        if len(matches) == 0:
            return []
        
        # Convert to the format expected by non-max suppression
        boxes = np.array([[x, y, x + w, y + h] for x, y, w, h in matches])
        scores = np.ones(len(matches))  # Equal weight to all matches
        
        # Perform non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.0, overlap_threshold)
        
        # Extract filtered matches
        if len(indices) > 0:
            if isinstance(indices, tuple) or (isinstance(indices, np.ndarray) and indices.ndim > 1):
                indices = indices.flatten()
                
            filtered_matches = [matches[i] for i in indices]
            return filtered_matches
        return []
    
    def _cluster_matches(self, points, distance_threshold=50):
        """Cluster matching points to identify separate instances of the object"""
        if len(points) == 0:
            return []
            
        # Flatten points array
        points = points.reshape(-1, 2)
        
        # Use DBSCAN clustering
        try:
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=distance_threshold, min_samples=3).fit(points)
            labels = clustering.labels_
            
            # Group points by cluster label
            clusters = []
            unique_labels = set(labels)
            for label in unique_labels:
                if label == -1:  # Skip noise points
                    continue
                cluster = points[labels == label]
                if len(cluster) >= 4:  # Only consider clusters with enough points
                    clusters.append(cluster)
            
            return clusters
        except ImportError:
            # Fall back to simpler clustering if sklearn is not available
            clusters = []
            for point in points:
                assigned = False
                for cluster in clusters:
                    # Check if point is close to any point in the cluster
                    for cluster_point in cluster:
                        dist = np.linalg.norm(point - cluster_point)
                        if dist < distance_threshold:
                            cluster.append(point)
                            assigned = True
                            break
                    if assigned:
                        break
                
                if not assigned:
                    # Create new cluster
                    clusters.append([point])
            
            return [np.array(c) for c in clusters if len(c) >= 4]
    
    def _highlight_in_pdf(self, doc, all_matches, output_path):
        """Highlight detected objects in the PDF and save to output_path"""
        # Create a copy of the document to avoid modifying the original
        output_doc = fitz.open()
        
        for page_num, matches, _ in all_matches:
            # Copy the page to the new document
            output_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
            page = output_doc[-1]  # Last inserted page
            
            for x, y, w, h in matches:
                # Create a rectangle annotation - coordinates already in PDF space
                rect = fitz.Rect(x, y, x+w, y+h)
                annot = page.add_rect_annot(rect)
                annot.set_colors(stroke=(0, 1, 0))  # Green border
                annot.set_border(width=1.5)  # Thicker border
                annot.update()
        
        # Copy any pages without matches
        processed_pages = set(page_num for page_num, _, _ in all_matches)
        for i in range(len(doc)):
            if i not in processed_pages:
                output_doc.insert_pdf(doc, from_page=i, to_page=i)
        
        # Save the document with annotations
        output_doc.save(output_path)
        output_doc.close()
        doc.close()
        
        return output_path


def detect_objects_in_cad(cad_folder, target_folder, output_folder, method='feature', threshold=0.7, dpi=300):
    """
    Detect objects in all CAD files matching targets and save results
    
    Args:
        cad_folder (str): Folder containing CAD PDF files
        target_folder (str): Folder containing target PNG images
        output_folder (str): Folder to save highlighted CAD files
        method (str): Detection method ('template', 'feature', 'contour')
        threshold (float): Matching threshold
        dpi (int): DPI for rendering PDF pages
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    detector = CADObjectDetector()
    
    # Get all CAD files
    cad_files = [f for f in os.listdir(cad_folder) if f.lower().endswith('.pdf')]
    
    # Get all target files
    target_files = [f for f in os.listdir(target_folder) if f.lower().endswith('.png')]
    
    if not cad_files:
        logger.warning(f"No PDF files found in {cad_folder}")
        return {}
    
    if not target_files:
        logger.warning(f"No PNG files found in {target_folder}")
        return {}
    
    results = {}
    
    for target_file in target_files:
        target_name = os.path.splitext(target_file)[0]
        target_path = os.path.join(target_folder, target_file)
        results[target_name] = {}
        
        logger.info(f"Processing target: {target_name}")
        
        for cad_file in cad_files:
            cad_name = os.path.splitext(cad_file)[0]
            cad_path = os.path.join(cad_folder, cad_file)
            output_path = os.path.join(output_folder, f"{cad_name}_{target_name}_detected.pdf")
            
            logger.info(f"Analyzing CAD file: {cad_name}")
            
            try:
                count, highlighted_path = detector.detect_objects(
                    cad_path, target_path, output_path, method, threshold, dpi)
                
                results[target_name][cad_name] = {
                    'count': count,
                    'highlighted_path': highlighted_path
                }
                
                logger.info(f"Found {count} instances of {target_name} in {cad_name}")
                if highlighted_path:
                    logger.info(f"Saved highlighted PDF at {highlighted_path}")
                
            except Exception as e:
                logger.error(f"Error processing {cad_path} with {target_path}: {e}")
    
    return results