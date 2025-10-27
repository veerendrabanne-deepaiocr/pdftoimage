import os
import fitz  # PyMuPDF
import docx
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

def standardize_to_pdf(file_path, original_filename, upload_folder):
    """
    Converts .docx and .txt files to a standardized PDF format using Python libraries.
    """
    file_ext = os.path.splitext(original_filename)[1].lower()
    pdf_filename = os.path.splitext(original_filename)[0] + '.pdf'
    pdf_filepath = os.path.join(upload_folder, pdf_filename)

    if file_ext == '.pdf':
        return file_path

    # Handle .docx files
    if file_ext == '.docx':
        doc = docx.Document(file_path)
        pdf = SimpleDocTemplate(pdf_filepath)
        styles = getSampleStyleSheet()
        story = []
        for para in doc.paragraphs:
            story.append(Paragraph(para.text, styles['Normal']))
            story.append(Spacer(1, 0.2 * inch))
        pdf.build(story)
        return pdf_filepath

    # Handle .txt files
    if file_ext == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            text_content = f.read()

        pdf = SimpleDocTemplate(pdf_filepath)
        styles = getSampleStyleSheet()
        story = [Paragraph(line, styles['Normal']) for line in text_content.splitlines()]
        pdf.build(story)
        return pdf_filepath

    # Explicitly reject .doc files
    if file_ext == '.doc':
        raise ValueError("Unsupported file type: .doc files are not supported in this version.")

    return None # Return None for any other unsupported types

def convert_pdf_to_images(pdf_path, original_filename, conversion_folder):
    """
    Converts a PDF file into a series of PNG images.
    """
    # Create a directory named after the original file
    output_folder_name = os.path.splitext(original_filename)[0]
    output_folder_path = os.path.join(conversion_folder, output_folder_name)
    os.makedirs(output_folder_path, exist_ok=True)

    image_paths = []

    # Open the PDF and convert each page to an image
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=150)

        image_filename = f'page_{page_num + 1}.png'
        image_path = os.path.join(output_folder_path, image_filename)
        pix.save(image_path)
        image_paths.append(image_path)

    return image_paths
