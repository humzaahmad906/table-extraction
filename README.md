# Table Extraction Application

## Overview
This is a simple Flask application. This application is used to extract table from an image using open-source models. The models that are used can be found [here](https://huggingface.co/docs/transformers/main/en/model_doc/table-transformer).

## Workflow
The basic workflow of the application is that an image gets uploaded to the api and then it goes through the detection table transformer first where we extract the table from the image, after that the cropped image goes through the structure table transformer to extract the bounding box of the table cells and finally we apply Tesseract OCR to each cell to get the text and postprocess the output into the required format

## Installation

Before you start, ensure you have met the following requirements:
* You have the linux os installed.
* You have installed Python 3.6+.
* You have installed pip.


Follow these steps to install the necessary Python packages and tesseract:

```bash
sudo apt install tesseract-ocr
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Usage
To run this application, navigate to the directory containing the application and use the following command:
```python
python main.py
```

The application will start a server on your machine, typically accessible at http://localhost:5000. 
Once the server is running, you can interact with it using HTTP requests. The specific endpoints, request formats, and other details are documented in [API documentation](https://documenter.getpostman.com/view/10524921/2sA2rAyMpH).