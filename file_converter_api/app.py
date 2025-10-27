import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import converter

# Configure application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['CONVERSION_FOLDER'] = 'conversions/'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'doc', 'docx', 'txt'}

# Ensure the upload and conversion folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['CONVERSION_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """Check if the file's extension is in the allowed set."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/convert', methods=['POST'])
def convert_file():
    """
    Handle file upload and conversion.
    """
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400

    file = request.files['file']

    # Check if the file has a name
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400

    # Check if the file type is allowed and save it
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Standardize the file to PDF
            pdf_path = converter.standardize_to_pdf(filepath, filename, app.config['UPLOAD_FOLDER'])

            if not pdf_path or not os.path.exists(pdf_path):
                return jsonify({"status": "error", "message": "Failed to convert to PDF"}), 500

            # Convert the PDF to images
            image_paths = converter.convert_pdf_to_images(pdf_path, filename, app.config['CONVERSION_FOLDER'])

            return jsonify({"status": "success", "image_paths": image_paths})

        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    return jsonify({"status": "error", "message": "File type not allowed"}), 400

if __name__ == '__main__':
    app.run(debug=True)
