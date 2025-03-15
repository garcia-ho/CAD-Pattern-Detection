import cv2
import numpy as np
import os
import multiprocessing
from joblib import Parallel, delayed
import concurrent.futures


class PatternDetector:
    def __init__(self, threshold=0.7, scale_range=(0.2, 2.0), scale_steps=20, rotations=None):
        """
        Initialize the pattern detector.
        
        Args:
            threshold: Matching threshold (0-1), higher values mean more strict matching
            scale_range: Tuple of (min_scale, max_scale) to search through
            scale_steps: Number of scale steps to use
            rotations: List of rotation angles to check (in degrees), defaults to [0, 90, 180, 270]
        """
        self.threshold = threshold
        self.scale_range = scale_range
        self.scale_steps = scale_steps
        self.rotations = rotations if rotations is not None else [0, 90, 180, 270]
        
    def detect_pattern(self, image, template):
        """
        Detect occurrences of the template in the image at multiple scales and rotations.
        
        Args:
            image: Input image to search in
            template: Template image to search for
            
        Returns:
            List of rectangles (x, y, w, h, angle) where pattern was found
        """
        # Convert images to grayscale
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
            
        if len(template.shape) == 3:
            gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            gray_template = template
        
        # Get dimensions
        img_h, img_w = gray_image.shape
        
        # Create scale list
        min_scale, max_scale = self.scale_range
        scales = np.linspace(min_scale, max_scale, self.scale_steps)
        
        # Store all matches from different scales and rotations
        all_rectangles = []
        scale_match_counts = {}
        
        # Try different rotations
        for angle in self.rotations:
            # Rotate template
            if angle == 0:
                rotated_template = gray_template
            else:
                # Rotate around center
                center = (gray_template.shape[1] // 2, gray_template.shape[0] // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated_template = cv2.warpAffine(
                    gray_template, 
                    rotation_matrix, 
                    (gray_template.shape[1], gray_template.shape[0]),
                    flags=cv2.INTER_LINEAR
                )
            
            # Generate scales for matching
            for scale in scales:
                # Resize template based on scale
                scaled_h = int(rotated_template.shape[0] * scale)
                scaled_w = int(rotated_template.shape[1] * scale)
                
                # Skip scales that would make the template larger than the image
                if scaled_h > img_h or scaled_w > img_w or scaled_h <= 10 or scaled_w <= 10:
                    continue
                    
                # Resize template
                scaled_template = cv2.resize(rotated_template, (scaled_w, scaled_h))
                
                # Perform template matching
                result = cv2.matchTemplate(gray_image, scaled_template, cv2.TM_CCOEFF_NORMED)
                
                # Find locations where the matching exceeds our threshold
                locations = np.where(result >= self.threshold)
                
                # Track matches per scale for anomaly detection
                scale_key = f"{angle}_{scale:.3f}"
                scale_match_counts[scale_key] = len(locations[0])
                
                # Convert to list of rectangles with rotation angle
                for pt in zip(*locations[::-1]):  # Swap columns and rows
                    all_rectangles.append((pt[0], pt[1], scaled_w, scaled_h, angle, scale))
                    
                # Print status for large-scale operations
                if len(locations[0]) > 0:
                    print(f"Found {len(locations[0])} matches at angle={angle}, scale={scale:.2f}")
        
        # Check for anomalies in match counts
        if scale_match_counts:
            avg_matches = sum(scale_match_counts.values()) / len(scale_match_counts)
            max_matches = max(scale_match_counts.values())
            
            # If any scale has significantly more matches than average, it might be detecting noise
            if max_matches > avg_matches * 5 and max_matches > 20:
                print(f"WARNING: Detected possible false positives. Max matches: {max_matches}, Avg: {avg_matches:.2f}")
                
                # Find the problematic scales
                problematic_scales = [k for k, v in scale_match_counts.items() if v > avg_matches * 3]
                print(f"Problematic scales: {problematic_scales}")
                
                # Filter out rectangles from problematic scales if necessary
                if max_matches > 100:  # If there's an extreme anomaly
                    filtered_rectangles = []
                    for rect in all_rectangles:
                        x, y, w, h, angle, scale = rect
                        scale_key = f"{angle}_{scale:.3f}"
                        if scale_key not in problematic_scales:
                            filtered_rectangles.append((x, y, w, h, angle))
                    
                    all_rectangles = filtered_rectangles
                else:
                    # Just remove the scale information
                    all_rectangles = [(x, y, w, h, angle) for (x, y, w, h, angle, scale) in all_rectangles]
            else:
                # Remove the scale information
                all_rectangles = [(x, y, w, h, angle) for (x, y, w, h, angle, scale) in all_rectangles]
        
        # Apply non-maximum suppression to avoid multiple detections of the same object
        filtered_rectangles = self._non_max_suppression(all_rectangles)
            
        return filtered_rectangles
    
    def _non_max_suppression(self, rectangles, overlap_threshold=0.3):
        """
        Apply non-maximum suppression to avoid duplicate detections
        
        Args:
            rectangles: List of rectangles (x, y, w, h, angle)
            overlap_threshold: Maximum allowed overlap
            
        Returns:
            Filtered list of rectangles
        """
        if not rectangles:
            return []
        
        # Group rectangles by angle
        angle_groups = {}
        for i, (x, y, w, h, angle) in enumerate(rectangles):
            if angle not in angle_groups:
                angle_groups[angle] = []
            angle_groups[angle].append((i, (x, y, w, h)))
        
        keep_indices = []
        
        # Process each angle group separately
        for angle, rects in angle_groups.items():
            indices = [r[0] for r in rects]
            boxes = np.array([[x, y, x + w, y + h] for (_, (x, y, w, h)) in rects])
            
            if boxes.size == 0:
                continue
            
            # Extract coordinates
            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = boxes[:, 2]
            y2 = boxes[:, 3]
            
            # Compute area of each box
            area = (x2 - x1) * (y2 - y1)
            
            # Reject extremely small areas (likely false positives)
            valid_indices = np.where(area > 100)[0]  # Minimum area threshold
            if valid_indices.size == 0:
                continue
                
            # Filter by area
            x1 = x1[valid_indices]
            y1 = y1[valid_indices]
            x2 = x2[valid_indices]
            y2 = y2[valid_indices]
            area = area[valid_indices]
            indices = [indices[i] for i in valid_indices]
            
            # Sort boxes by area (larger first)
            order = np.argsort(area)[::-1]
            
            keep = []
            while order.size > 0:
                i = order[0]
                keep.append(i)
                
                # Find the intersection with remaining boxes
                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])
                
                # Compute width and height of intersection
                w = np.maximum(0, xx2 - xx1)
                h = np.maximum(0, yy2 - yy1)
                
                # Compute IoU
                intersection = w * h
                union = area[i] + area[order[1:]] - intersection
                iou = intersection / (union + 1e-6)
                
                # Higher overlap threshold for smaller objects
                adaptive_threshold = overlap_threshold
                if area[i] < 500:  # For small objects
                    adaptive_threshold = 0.2  # Stricter threshold
                
                # Remove indices with IoU > threshold
                inds = np.where(iou <= adaptive_threshold)[0]
                order = order[inds + 1]
                
            # Map back to original indices
            keep_indices.extend([indices[k] for k in keep])
        
        # Now do another pass to remove overlaps between different angles
        if len(keep_indices) > 1:
            final_rectangles = [rectangles[i] for i in keep_indices]
            all_boxes = np.array([[x, y, x + w, y + h] 
                                for (x, y, w, h, _) in final_rectangles])
            
            x1 = all_boxes[:, 0]
            y1 = all_boxes[:, 1]
            x2 = all_boxes[:, 2]
            y2 = all_boxes[:, 3]
            
            area = (x2 - x1) * (y2 - y1)
            
            # Sort by area again (larger first)
            order = np.argsort(area)[::-1]
            
            final_keep = []
            while order.size > 0:
                i = order[0]
                final_keep.append(i)
                
                # Skip if this is the last box
                if order.size == 1:
                    break
                    
                # Calculate IoU with remaining boxes
                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])
                
                # Compute width and height of intersection
                w = np.maximum(0, xx2 - xx1)
                h = np.maximum(0, yy2 - yy1)
                
                # Compute IoU
                intersection = w * h
                union = area[i] + area[order[1:]] - intersection
                iou = intersection / (union + 1e-6)
                
                # Dynamic threshold based on area
                adaptive_threshold = overlap_threshold
                if area[i] < 500:
                    adaptive_threshold = 0.2
                
                # Remove indices with IoU > threshold
                inds = np.where(iou <= adaptive_threshold)[0]
                order = order[inds + 1]
                
            # Return final filtered rectangles
            return [final_rectangles[i] for i in final_keep]
        
        # Return filtered rectangles with their angles
        return [rectangles[i] for i in keep_indices]
    
    def highlight_matches(self, image, matches, color=(0, 0, 255), thickness=2):
        """
        Highlight the matches in the image using parallel processing.
        
        Args:
            image: Input image
            matches: List of rectangles (x, y, w, h, angle)
            color: Color to use for highlighting (BGR)
            thickness: Line thickness
            
        Returns:
            Image with highlighted matches
        """
        result = image.copy()
        
        # Set number of cores to use
        num_cores = 16
        
        # If only a few matches, don't use parallelization
        if len(matches) <= 4:
            for match in matches:
                result = self._highlight_single_match(result, match, color, thickness)
            return result
            
        # Process matches in parallel using joblib
        processed_results = Parallel(n_jobs=num_cores)(
            delayed(self._highlight_single_match)(result.copy(), match, color, thickness) 
            for match in matches
        )
        
        # Combine the results
        if processed_results:
            # Start with the original image
            final_result = image.copy()
            
            # Combine all processed images
            for proc_result in processed_results:
                # Create a mask where the processed result differs from the original
                mask = cv2.absdiff(proc_result, image) > 0
                # Apply those differences to the final result
                final_result = np.where(mask, proc_result, final_result)
            
            return final_result
        else:
            return result
    
    def _highlight_single_match(self, image, match, color=(0, 0, 255), thickness=2):
        """
        Highlight a single match in the image.
        
        Args:
            image: Input image
            match: Tuple (x, y, w, h, angle)
            color: Color to use for highlighting (BGR)
            thickness: Line thickness
            
        Returns:
            Image with the match highlighted
        """
        result = image.copy()
        x, y, w, h, angle = match
        
        if angle % 180 == 0:  # For 0 and 180 degrees, use normal rectangle
            # Create a mask for the region
            mask = np.zeros_like(result)
            cv2.rectangle(mask, (x, y), (x + w, y + h), color, -1)  # -1 means filled
            
            # Apply semi-transparent overlay
            alpha = 0.5  # Transparency factor
            result = cv2.addWeighted(result, 1, mask, alpha, 0)
            
            # Draw rectangle outline
            cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
            
            # Add rotation text
            cv2.putText(result, f"{angle}°", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 1)
        else:
            # For rotated rectangles, use a rotated rectangle representation
            rect = ((x + w/2, y + h/2), (w, h), angle)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # Create mask and fill polygon
            mask = np.zeros_like(result)
            cv2.fillPoly(mask, [box], color)
            
            # Apply semi-transparent overlay
            alpha = 0.5
            result = cv2.addWeighted(result, 1, mask, alpha, 0)
            
            # Draw polygon outline
            cv2.drawContours(result, [box], 0, color, thickness)
            
            # Add rotation text
            cv2.putText(result, f"{angle}°", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 1)
        
        return result
        
    # Alternative implementation using concurrent.futures
    def highlight_matches_alt(self, image, matches, color=(0, 0, 255), thickness=2):
        """
        Highlight the matches in the image using concurrent.futures.
        
        Args:
            image: Input image
            matches: List of rectangles (x, y, w, h, angle)
            color: Color to use for highlighting (BGR)
            thickness: Line thickness
            
        Returns:
            Image with highlighted matches
        """
        result = image.copy()
        
        # Set number of cores to use
        num_cores = 16
        
        # If only a few matches, don't use parallelization
        if len(matches) <= 4:
            for match in matches:
                result = self._highlight_single_match(result, match, color, thickness)
            return result
        
        # Process matches in parallel using concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
            # Submit all tasks and store futures
            futures = [executor.submit(self._highlight_single_match, 
                                     image.copy(), match, color, thickness) 
                     for match in matches]
            
            # Collect results as they complete
            processed_results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Combine the results
        if processed_results:
            # Start with the original image
            final_result = image.copy()
            
            # Combine all processed images
            for proc_result in processed_results:
                # Create a mask where the processed result differs from the original
                mask = cv2.absdiff(proc_result, image) > 0
                # Apply those differences to the final result
                final_result = np.where(mask, proc_result, final_result)
            
            return final_result
        else:
            return result
    
    def draw_match_count(self, image, count):
        """
        Draw match count on the image.
        
        Args:
            image: Input image
            count: Number of matches
            
        Returns:
            Image with text showing match count
        """
        result = image.copy()
        text = f"Matches: {count}"
        cv2.putText(result, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return result