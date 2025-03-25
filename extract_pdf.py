import os
import pdfplumber

def extract_text(pdf_path):
    """
    Extracts text from a single PDF file using pdfplumber.
    Returns the extracted text as a string.
    """
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

if __name__ == "__main__":
    pdf_folder = "pdfs"
    output_folder = "extracted_texts"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            text = extract_text(pdf_path)
            output_file = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Extracted text from '{filename}' and saved to '{output_file}'.")
