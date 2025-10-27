import os
import subprocess
import fitz  # PyMuPDF

def standardize_to_pdf(file_path, original_filename, upload_folder):
    """
    Converts a variety of file types to a standardized PDF format.
    """
    file_ext = os.path.splitext(original_filename)[1].lower()

    if file_ext == '.pdf':
        return file_path

    if file_ext in ['.doc', '.docx', '.txt']:
        # For this to work, LibreOffice must be installed on the server.
        # The command converts the file to PDF and places it in the upload_folder.
        subprocess.run([
            'soffice', '--headless', '--convert-to', 'pdf',
            file_path, '--outdir', upload_folder
        ], check=True)

        pdf_filename = os.path.splitext(original_filename)[0] + '.pdf'
        return os.path.join(upload_folder, pdf_filename)

    return None

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
