from langchain.text_splitter import RecursiveCharacterTextSplitter
from pdf2image import convert_from_path
from pytesseract import image_to_string
from PIL import Image
import re
from fpdf import FPDF

# Set up paths
PDF_PATH = "C:\Desktop\Clean PDF\manuals\exon.pdf"  # Path to your input PDF
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Path to Tesseract
OUTPUT_TEXT_FILE = "exon.txt"  # Output text file


# Ensure pytesseract uses the correct path
import pytesseract
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Function to extract text from a PDF using OCR
def extract_text_from_pdf(pdf_path):
    print("Converting PDF to images...")
    images = convert_from_path(pdf_path)  # Convert PDF pages to images
    full_text = ""

    for page_number, image in enumerate(images, start=1):
        print(f"Processing page {page_number}...")
        text = image_to_string(image)  # Extract text from image
        full_text += f"\n--- Page {page_number} ---\n{text}"

    return full_text

# Function to clean the extracted text
def clean_text(text):
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Optional: Remove page markers or unwanted patterns
    text = re.sub(r'--- Page \d+ ---', '', text)
    return text.strip()

# Use LangChain for advanced text processing (e.g., splitting into chunks)
def structure_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    structured_text = splitter.split_text(text)
    return structured_text


# Main script
if __name__ == "__main__":
    # Step 1: Extract text from the PDF
    print("Starting text extraction...")
    raw_text = extract_text_from_pdf(PDF_PATH)

    # Step 2: Clean the extracted text
    print("Cleaning extracted text...")
    cleaned_text = clean_text(raw_text)

    # Step 3: Use LangChain to structure text
    print("Structuring text into chunks...")
    structured_chunks = structure_text(cleaned_text)

    # Step 4: Save the cleaned and structured text to a plain text file
    print(f"Saving cleaned text to {OUTPUT_TEXT_FILE}...")
    with open(OUTPUT_TEXT_FILE, "w", encoding="utf-8") as file:
        for chunk in structured_chunks:
            file.write(chunk + "\n\n")
    print(f"Cleaned and structured text saved to {OUTPUT_TEXT_FILE}.")


    print("Processing complete!")
