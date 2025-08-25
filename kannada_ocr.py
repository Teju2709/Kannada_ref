#!/usr/bin/env python3
"""
Kannada PDF OCR with Image Enhancement
Converts old, low-contrast Kannada PDFs to editable text
"""

import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
import os
from PIL import Image, ImageEnhance, ImageFilter
import argparse
# ...existing code...
import api  # This will set up the API key
import google.generativeai as genai

# Now you can use genai as needed
# ...existing code...

class KannadaPDFOCR:
    def __init__(self, tesseract_path=None):
        """
        Initialize the OCR processor
        Args:
            tesseract_path: Path to tesseract executable (if not in PATH)
        """
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Tesseract configuration for Kannada
        self.config = r'--oem 3 --psm 6 -l kan'

    def split_columns(self, pil_image):
        """
        Detect vertical columns and split the image into column images.
        Returns a list of PIL Images, one per column (left to right).
        """
        img = np.array(pil_image.convert("L"))  # Ensure grayscale
        # Threshold to binary
        _, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
        # Detect vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, img.shape[0] // 30))
        vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        # Find contours of vertical lines
        contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        column_boxes = [cv2.boundingRect(cnt) for cnt in contours]
        column_boxes = sorted(column_boxes, key=lambda x: x[0])  # Sort left to right

        # If no columns detected, return the whole image
        if not column_boxes or len(column_boxes) == 1:
            return [pil_image]

        # Split image into columns
        columns = []
        for x, y, w, h in column_boxes:
            if w > 10:  # Filter out thin lines
                col_img = pil_image.crop((x, 0, x + w, pil_image.height))
                columns.append(col_img)
        # If nothing found, fallback to whole image
        if not columns:
            return [pil_image]
        return columns


    
    
    def enhance_image(self, image, enhancement_level='medium'):
        """
        Enhance image quality for better OCR results
        
        Args:
            image: PIL Image object
            enhancement_level: 'light', 'medium', 'heavy'
        
        Returns:
            Enhanced PIL Image
        """
        # Convert to numpy array for OpenCV operations
        img_array = np.array(image)
        
        # Convert to grayscale if not already
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Enhancement based on level
        if enhancement_level == 'light':
            enhanced = self._light_enhancement(gray)
        elif enhancement_level == 'medium':
            enhanced = self._medium_enhancement(gray)
        else:  # heavy
            enhanced = self._heavy_enhancement(gray)
        
        # Convert back to PIL Image
        return Image.fromarray(enhanced)
    
    def _light_enhancement(self, gray):
        """Light enhancement for slightly faded documents"""
        # Histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Mild sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced
    
    def _medium_enhancement(self, gray):
        """Medium enhancement for moderately degraded documents"""
        # Denoise first
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Gamma correction
        gamma = 1.2
        enhanced = np.power(enhanced/255.0, gamma) * 255
        enhanced = enhanced.astype(np.uint8)
        
        # Sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        # Threshold to create clean binary image
        _, enhanced = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return enhanced
    
    def _heavy_enhancement(self, gray):
        """Heavy enhancement for severely degraded documents"""
        # Strong denoising
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        cleaned = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
        
        # Strong contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
        enhanced = clahe.apply(cleaned)
        
        # Gamma correction
        gamma = 0.8
        enhanced = np.power(enhanced/255.0, gamma) * 255
        enhanced = enhanced.astype(np.uint8)
        
        # Bilateral filter to smooth while preserving edges
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Strong sharpening
        kernel = np.array([[-1,-1,-1,-1,-1],
                          [-1, 2, 2, 2,-1],
                          [-1, 2, 8, 2,-1],
                          [-1, 2, 2, 2,-1],
                          [-1,-1,-1,-1,-1]]) / 8.0
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        # Final thresholding
        _, enhanced = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return enhanced
    
    def pdf_to_images(self, pdf_path, dpi=300):
        """
        Convert PDF pages to images
        
        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for conversion (higher = better quality)
        
        Returns:
            List of PIL Images
        """
        try:
            images = convert_from_path(pdf_path, dpi=dpi)
            return images
        except Exception as e:
            print(f"Error converting PDF: {e}")
            return []
    
    def ocr_image(self, image):
        """
        Perform OCR on a single image
        
        Args:
            image: PIL Image object
        
        Returns:
            Extracted text string
        """
        try:
            text = pytesseract.image_to_string(image, config=self.config)
            return text
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""
    
    def process_pdf(self, pdf_path, output_path, enhancement_level='medium', save_enhanced_images=False):
        """
        Process entire PDF and extract Kannada text
        
        Args:
            pdf_path: Input PDF file path
            output_path: Output text file path
            enhancement_level: Image enhancement level
            save_enhanced_images: Whether to save enhanced images
        
        Returns:
            Success status
        """
        print(f"Processing PDF: {pdf_path}")
        
        # Convert PDF to images
        images = self.pdf_to_images(pdf_path)
        if not images:
            return False
        
        print(f"Converted to {len(images)} images")
        
        # Create output directory for enhanced images if needed
        if save_enhanced_images:
            img_dir = os.path.splitext(output_path)[0] + "_enhanced_images"
            os.makedirs(img_dir, exist_ok=True)
        
        all_text = []
        
        for i, image in enumerate(images):
            print(f"Processing page {i+1}/{len(images)}")
            
            # Enhance image
            enhanced_image = self.enhance_image(image, enhancement_level)
            
            # Save enhanced image if requested
            if save_enhanced_images:
                enhanced_image.save(os.path.join(img_dir, f"page_{i+1:03d}.png"))
            
            # Perform OCR
            text = self.ocr_image(enhanced_image)
            all_text.append(f"--- Page {i+1} ---\n{text}\n")
        
        # Save extracted text
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(all_text))
            print(f"Text saved to: {output_path}")
            return True
        except Exception as e:
            print(f"Error saving text: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Convert Kannada PDF to text using OCR')
    parser.add_argument('input_pdf', help='Input PDF file path')
    parser.add_argument('output_text', help='Output text file path')
    parser.add_argument('--enhancement', choices=['light', 'medium', 'heavy'], 
                       default='medium', help='Image enhancement level')
    parser.add_argument('--save-images', action='store_true', 
                       help='Save enhanced images')
    parser.add_argument('--tesseract-path', help='Path to tesseract executable')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for PDF conversion')
    
    args = parser.parse_args()
    
    # Initialize OCR processor
    ocr_processor = KannadaPDFOCR(args.tesseract_path)
    
    # Process PDF
    success = ocr_processor.process_pdf(
        args.input_pdf, 
        args.output_text, 
        args.enhancement,
        args.save_images
    )
    
    if success:
        print("PDF processing completed successfully!")
    else:
        print("PDF processing failed!")

if __name__ == "__main__":
    # Example usage without command line
    if len(os.sys.argv) == 1:  # No command line arguments
        # Example configuration
        pdf_path = "karmaveera4march0000unse.pdf"
        output_path = "extracted_text.txt"
        
        # Create OCR processor
        ocr = KannadaPDFOCR()
        
        # Process PDF with medium enhancement
        ocr.process_pdf(pdf_path, output_path, enhancement_level='medium', save_enhanced_images=True)
    else:
        main()
        