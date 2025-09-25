# src/app/model.py
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from skimage.transform import resize
from PIL import Image

# ---------------------------
# U-Net
# ---------------------------
def unet_model(input_size=(256,256,1)):
    inputs = tf.keras.Input(shape=input_size)
    def conv_block(x, n_filters):
        x = layers.Conv2D(n_filters, 3, activation="relu", padding="same")(x)
        x = layers.Conv2D(n_filters, 3, activation="relu", padding="same")(x)
        return x
    c1 = conv_block(inputs,16)
    p1 = layers.MaxPooling2D((2,2))(c1)
    c2 = conv_block(p1,32)
    p2 = layers.MaxPooling2D((2,2))(c2)
    c3 = conv_block(p2,64)
    p3 = layers.MaxPooling2D((2,2))(c3)
    b = conv_block(p3,128)
    u3 = layers.Conv2DTranspose(64,2,strides=2,padding="same")(b)
    u3 = layers.Concatenate()([u3, c3])
    c4 = conv_block(u3,64)
    u2 = layers.Conv2DTranspose(32,2,strides=2,padding="same")(c4)
    u2 = layers.Concatenate()([u2,c2])
    c5 = conv_block(u2,32)
    u1 = layers.Conv2DTranspose(16,2,strides=2,padding="same")(c5)
    u1 = layers.Concatenate()([u1,c1])
    c6 = conv_block(u1,16)
    outputs = layers.Conv2D(1,1,activation="sigmoid")(c6)
    model = models.Model(inputs,outputs)
    return model

# ---------------------------
# Overlay
# ---------------------------
def overlay_mask_on_image(orig_image_pil, mask, alpha=0.4):
    orig = np.array(orig_image_pil).astype(np.uint8)
    if orig.ndim == 2:
        orig_rgb = np.stack([orig]*3, axis=-1)
    elif orig.ndim == 3 and orig.shape[2] == 4:
        orig_rgb = orig[:,:,:3]
    else:
        orig_rgb = orig.copy()
    mask_resized = resize(mask, (orig_rgb.shape[0], orig_rgb.shape[1]), order=0, preserve_range=True, anti_aliasing=False)
    mask_bin = (mask_resized>0.5).astype(np.uint8)
    overlay = orig_rgb.copy()
    overlay[mask_bin==1] = [255,0,0]
    blended = ((1-alpha)*orig_rgb + alpha*overlay).astype(np.uint8)
    return Image.fromarray(blended)
