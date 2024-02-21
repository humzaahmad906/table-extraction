import torch
from torchvision import transforms
import numpy as np
from tqdm.auto import tqdm

from transformers import AutoModelForObjectDetection, TableTransformerForObjectDetection
from PIL import Image
import pytesseract
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_models():
    """
    Get the models
    :return: (table detection model, structure detection model)
    """
    model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
    model.to(device)
    model_id = "microsoft/table-structure-recognition-v1.1-all"
    structure_model = TableTransformerForObjectDetection.from_pretrained(model_id)
    structure_model.to(device)
    return model, structure_model


class MaxResize(object):
    """
    Set the size of the image to a max size
    """

    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize((int(round(scale * width)), int(round(scale * height))))

        return resized_image


def get_transforms():
    """
    Get the transforms to be applied for preprocessing of images before passing it through the model
    :return:
    """
    detection_transform = transforms.Compose([
        MaxResize(800),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    structure_transform = transforms.Compose([
        MaxResize(1000),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return detection_transform, structure_transform


def box_cxcywh_to_xyxy(x):
    """
    Format the bounding boxes from (center, height, width) format to (left, top, right, bottom) format
    :param x: (center x, center y, height, width) format torch tensor
    :return: (left, top, right, bottom) format torch tensor
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    """
    Rescale bounding boxes to the size of the original size
    :param out_bbox: (left, top, right, bottom) format boxes scaled from 0 to 1 tensor
    :param size: original width and the height of the image that is passed through the model
    :return: (left, top, right, bottom) format boxes scaled to the original size tensor
    """
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def outputs_to_objects(outputs, img_size, id2label):
    """
    Post processing of the bounding boxes to dictionaries
    """
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if not class_label == 'no object':
            objects.append({'label': class_label, 'score': float(score),
                            'bbox': [float(elem) for elem in bbox]})

    return objects


def get_cell_coordinates_by_row(table_data):
    """
    Extract cell coordinates based on row and column format
    """
    # Extract rows and columns
    rows = [entry for entry in table_data if entry['label'] == 'table row']
    columns = [entry for entry in table_data if entry['label'] == 'table column']

    # Sort rows and columns by their Y and X coordinates, respectively
    rows.sort(key=lambda x: x['bbox'][1])
    columns.sort(key=lambda x: x['bbox'][0])

    # Function to find cell coordinates
    def find_cell_coordinates(row, column):
        cell_bbox = [column['bbox'][0], row['bbox'][1], column['bbox'][2], row['bbox'][3]]
        return cell_bbox

    # Generate cell coordinates and count cells in each row
    cell_coordinates = []

    for row in rows:
        row_cells = []
        for column in columns:
            cell_bbox = find_cell_coordinates(row, column)
            row_cells.append({'column': column['bbox'], 'cell': cell_bbox})

        # Sort cells in the row by X coordinate
        row_cells.sort(key=lambda x: x['column'][0])

        # Append row information to cell_coordinates
        cell_coordinates.append({'row': row['bbox'], 'cells': row_cells, 'cell_count': len(row_cells)})

    # Sort rows from top to bottom
    cell_coordinates.sort(key=lambda x: x['row'][1])

    return cell_coordinates


def apply_ocr(cell_coordinates, cropped_table):
    """
    Apply OCR to each cell coordinate and resturn as json
    """

    data = []
    max_num_columns = 0
    for idx, row in enumerate(tqdm(cell_coordinates)):
        row_text = []
        for cell in row["cells"]:
            cell_image = np.array(cropped_table.crop(cell["cell"]))
            custom_config = r'--oem 3 --psm 6'
            result = pytesseract.image_to_string(np.array(cell_image), config=custom_config)
            result = result.replace("|", "").strip()
            row_text.append(result)
        data.append(row_text)

    return data


def objects_to_crops(img, tokens, objects, class_thresholds, padding=10):
    """
    Process the bounding boxes produced by the table detection model into
    cropped table images and cropped tokens.
    """

    table_crops = []
    for obj in objects:
        if obj['score'] < class_thresholds[obj['label']]:
            continue

        cropped_table = {}

        bbox = obj['bbox']
        bbox = [bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding]

        cropped_img = img.crop(bbox)

        table_tokens = [token for token in tokens if iob(token['bbox'], bbox) >= 0.5]
        for token in table_tokens:
            token['bbox'] = [token['bbox'][0] - bbox[0],
                             token['bbox'][1] - bbox[1],
                             token['bbox'][2] - bbox[0],
                             token['bbox'][3] - bbox[1]]

        # If table is predicted to be rotated, rotate cropped image and tokens/words:
        if obj['label'] == 'table rotated':
            cropped_img = cropped_img.rotate(270, expand=True)
            for token in table_tokens:
                bbox = token['bbox']
                bbox = [cropped_img.size[0] - bbox[3] - 1,
                        bbox[0],
                        cropped_img.size[0] - bbox[1] - 1,
                        bbox[2]]
                token['bbox'] = bbox

        cropped_table['image'] = cropped_img
        cropped_table['tokens'] = table_tokens

        table_crops.append(cropped_table)

    return table_crops


def get_crop_table(file_path, model, detection_transform):
    """
    Apply detection model to the image to crop the table out of the document
    """
    image = Image.open(file_path).convert("RGB")
    pixel_values = detection_transform(image).unsqueeze(0)
    pixel_values = pixel_values.to(device)
    with torch.no_grad():
        outputs = model(pixel_values)

    id2label = model.config.id2label
    id2label[len(model.config.id2label)] = "no object"
    objects = outputs_to_objects(outputs, image.size, id2label)
    tokens = []
    detection_class_thresholds = {
        "table": 0.5,
        "table rotated": 0.5,
        "no object": 10
    }
    crop_padding = 10

    tables_crops = objects_to_crops(image, tokens, objects, detection_class_thresholds, padding=crop_padding)
    cropped_table = tables_crops[0]['image'].convert("RGB")
    return cropped_table


def get_cells_json(cropped_table, structure_model, structure_transform):
    """
    Apply structure model to the cropped table to extract all cells in format of json
    after applying Tesseract-OCR on each cell
    """
    pixel_values = structure_transform(cropped_table).unsqueeze(0)
    pixel_values = pixel_values.to(device)
    with torch.no_grad():
        outputs = structure_model(pixel_values)

    structure_id2label = structure_model.config.id2label
    structure_id2label[len(structure_id2label)] = "no object"

    cells = outputs_to_objects(outputs, cropped_table.size, structure_id2label)
    cell_coordinates = get_cell_coordinates_by_row(cells)
    data = apply_ocr(cell_coordinates, cropped_table)
    df = pd.DataFrame(columns=data[0])
    for row in data[1:]:
        df.loc[len(df)] = row
    list_of_dicts = df.to_dict('records')
    return list_of_dicts
