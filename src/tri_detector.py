import cv2
import numpy as np


class TriangleDetector:
    def __init__(self):
        self.min_area = 100  # Minimum area of triangle to be detected
        self.approx_epsilon_factor = 0.02  # Approximation accuracy

    def detect_triangles(self, image):
        """
        Detect triangles in the given image.
        
        Args:
            image: Input image (BGR)
            
        Returns:
            List of triangles (each triangle is a numpy array of 3 points)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold or edge detection
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        triangles = []
        
        # Iterate through each contour
        for contour in contours:
            # Filter small contours
            if cv2.contourArea(contour) < self.min_area:
                continue
                
            # Approximate contour with polygon
            epsilon = self.approx_epsilon_factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's a triangle (has 3 vertices)
            if len(approx) == 3:
                triangles.append(approx)
                
        return triangles

    def fill_triangles(self, image, triangles, color=(0, 0, 255)):  # Red color (BGR)
        """
        Fill detected triangles with a specific color
        
        Args:
            image: Input image
            triangles: List of triangles
            color: Color to fill triangles with (BGR format)
            
        Returns:
            Image with filled triangles
        """
        # Create a copy of the image to avoid modifying the original
        result = image.copy()
        
        # Draw filled triangles
        cv2.fillPoly(result, triangles, color)
        
        return result

    def draw_triangle_count(self, image, count):
        """
        Draw triangle count on the image
        
        Args:
            image: Input image
            count: Number of triangles detected
            
        Returns:
            Image with text indicating triangle count
        """
        result = image.copy()
        text = f"Triangles: {count}"
        cv2.putText(result, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        return result