from flask import Flask, request, jsonify
import os

from model import get_models, get_transforms, get_crop_table, get_cells_json

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

detection_model, structure_model = get_models()
detection_transform, structure_transform = get_transforms()

app = Flask(__name__)


def allowed_file(filename):
    """
    Check if file is an image (in the allowed extensions)
    """
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    if file:
        filename = file.filename

        if allowed_file(filename):
            file_path = os.path.join('uploads', filename)
            file.save(file_path)
            cropped_table = get_crop_table(file_path, detection_model, detection_transform)
            list_of_dicts = get_cells_json(cropped_table, structure_model, structure_transform)
            return jsonify(list_of_dicts), 200
        else:
            return jsonify({"error": "File uploaded is not an image"}), 400


if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
