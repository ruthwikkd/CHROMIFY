# Chromify: AI-Powered Image Colorization

Chromify is a deep learning-powered tool that automatically colorizes black and white images. This project uses a convolutional neural network (CNN) to predict and apply colors to grayscale images, bringing old photos and monochrome pictures to life.

## Features

- **Automatic Colorization**: Instantly add color to black and white images.
- **Web Interface**: Easy-to-use web interface for uploading and colorizing images.
- **Deep Learning Model**: Utilizes a U-Net-like architecture for accurate color prediction.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

You'll need Python 3 and pip installed on your system. You can install the required Python packages using the following command:

```bash
pip install -r requirements.txt
```
**Note:** A `requirements.txt` file is not yet available. This will be added in a future update. For now, you can install the dependencies by inspecting the imports in the python files.

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd <project-directory>
   ```
3. Run the application:
   ```bash
   python exp/app.py
   ```
This will start the Gradio web server. You can access the application by navigating to the URL provided in the terminal (usually `http://127.0.0.1:7860`).

## Usage

1. Open the web interface in your browser.
2. Upload a black and white image using the "Input" box.
3. The colorized image will be displayed in the "Output" box.

## Model

The colorization model is a U-Net-like convolutional neural network built with Keras. It's trained on a large dataset of color images to learn the mapping from grayscale to color. The model works in the LAB color space, where 'L' represents lightness, and 'a' and 'b' represent the color channels. The model takes the 'L' channel as input and predicts the 'a' and 'b' channels. The predicted color channels are then combined with the original lightness channel to produce the final colorized image.

The model architecture and training process can be found in the `BETA VERSION/beta_version.ipynb` notebook. The Gradio application for inference is in `exp/app.py`.
