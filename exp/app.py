from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers import BatchNormalization
from keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab
from skimage.io import imsave
import numpy as np
import os
import random
import tensorflow as tf
from PIL import Image, ImageTk
import cv2
import gradio as gr
import time
from skimage import img_as_ubyte

# Custom directories for input and output images
input_dir = "custom_input"
output_dir = "custom_output"
os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

def upscaler(img_path):
    try:
        # Save the input image to the custom directory
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image from {img_path}")
            return None, None
        input_image_path = os.path.join(input_dir, "input_image.png")
        cv2.imwrite(input_image_path, img)

        # Open the image and convert to RGB
        image = Image.open(img_path).convert("RGB")
        
        # Crop the image to a square by taking the center
        width, height = image.size
        min_side = min(width, height)
        left = (width - min_side) // 2
        top = (height - min_side) // 2
        right = left + min_side
        bottom = top + min_side
        cropped_image = image.crop((left, top, right, bottom))
        
        # Resize the cropped square to 400x400
        cropped_image = cropped_image.resize((400, 400))
        captured_image = np.array(cropped_image)
        
        # Save the preprocessed image to the input directory
        input_image_path = os.path.join(input_dir, "input_image.png")
        Image.fromarray(captured_image).save(input_image_path)
        
        # Load the preprocessed image
        image = img_to_array(load_img('custom_input/input_image.png'))
        image = np.array(image, dtype=float)

        X = rgb2lab(1.0/255*image)[:,:,0]
        Y = rgb2lab(1.0/255*image)[:,:,1:]
        Y /= 128
        X = X.reshape(1, 400, 400, 1)
        Y = Y.reshape(1, 400, 400, 2)

        # Building the neural network
        model = Sequential()
        model.add(InputLayer(input_shape=(None, None, 1)))
        model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
        model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))

        # Finish model
        model.compile(optimizer='rmsprop',loss='mse')

        # Ensure the model fits the data correctly
        model.fit(x=X, 
                  y=Y,
                  batch_size=1,
                  epochs=1000)
        

        output = model.predict(X)
        output *= 128  # Scale the output as per LAB color space requirements

        # Convert output to image format and save
        cur = np.zeros((400, 400, 3))
        cur[:, :, 0] = X[0][:, :, 0]  # Lightness channel from the input
        cur[:, :, 1:] = output[0]     # Predicted color channels

        # Ensure the results directory exists
        output_dir = "custom_output"
        os.makedirs(output_dir, exist_ok=True)

        # Convert to uint8 before saving
        cur_uint8 = img_as_ubyte(lab2rgb(cur))
        imsave(os.path.join(output_dir, "img_result.png"), cur_uint8)

        # Convert grayscale version to uint8 and save
        gray_version = img_as_ubyte(rgb2gray(lab2rgb(cur)))
        imsave(os.path.join(output_dir, "img_gray_version.png"), gray_version)

        # Wait for the output images to be available
        output_image_1_path = os.path.join(output_dir, "img_gray_version.png")
        output_image_2_path = os.path.join(output_dir, "img_result.png")

        # Load the output images and return them
        output_image_1 = cv2.cvtColor(cv2.imread(output_image_1_path), cv2.COLOR_BGR2RGB)
        output_image_2 = cv2.cvtColor(cv2.imread(output_image_2_path), cv2.COLOR_BGR2RGB)
        return output_image_1, output_image_2
    except Exception as error:
        print("Error:", error)
        return None, None

if __name__ == "__main__":
    title = "Chromify"

    demo = gr.Interface(
        fn=upscaler,
        inputs=gr.Image(type="filepath", label="Input"),
        outputs=[
            gr.Image(type="numpy", label="Output Image 1"),
            gr.Image(type="numpy", label="Output Image 2"),
        ],
        title=title,
        allow_flagging="never",
    )

    demo.queue()
    demo.launch()
