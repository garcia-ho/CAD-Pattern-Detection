import PyPDF2
import os
import argparse

def extract_text_from_pdf(pdf_path, output_txt_path=None):
    """
    Extract text from a PDF file and save it to a text file.
    
    Args:
        pdf_path (str): Path to the PDF file.
        output_txt_path (str, optional): Path to save the extracted text.
                                         If None, will use PDF name with .txt extension.
    
    Returns:
        str: Path to the output text file.
    """
    # Default output path if not specified
    if output_txt_path is None:
        base_name = os.path.splitext(pdf_path)[0]
        output_txt_path = f"{base_name}.txt"
    
    try:
        # Open the PDF file
        with open(pdf_path, 'rb') as pdf_file:
            # Create a PDF reader object
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Get total number of pages
            num_pages = len(pdf_reader.pages)
            
            # Extract text from each page
            all_text = ""
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                all_text += page.extract_text() + "\n\n"
            
            # Write the extracted text to output file
            with open(output_txt_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(all_text)
            
            print(f"Text extracted successfully from '{pdf_path}'")
            print(f"Text saved to '{output_txt_path}'")
            
            return output_txt_path
    
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Extract text from PDF files")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("-o", "--output", help="Path to save the output text file (optional)")
    
    args = parser.parse_args()
    
    # Extract text from PDF
    extract_text_from_pdf(args.pdf_path, args.output)