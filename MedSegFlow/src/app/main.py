# main.py
### MedSegFlow Custom CNN + Heatmap + Contour (Grayscale MRI)

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import os
import zipfile
import io
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from skimage import measure

# ---------------------------
# Βασικές ρυθμίσεις
# ---------------------------
st.set_page_config(page_title="MedSegFlow CNN Analyzer", layout="centered")
st.markdown("<h1 style='text-align: center; color: rgb(200,200,125);'>🧠 MedSegFlow Custom CNN Analyzer (YES/NO)</h1>", unsafe_allow_html=True)

# ---------------------------
# Helper functions
# ---------------------------
def extract_zip_to_folder(zip_bytes, original_zip_name="dataset_temp"):
    folder_name = os.path.splitext(original_zip_name)[0]
    if not os.path.exists(folder_name):
        os.makedirs(folder_name, exist_ok=True)
    z = zipfile.ZipFile(io.BytesIO(zip_bytes))
    z.extractall(folder_name)
    return folder_name

def create_image_generators(folder, target_size=(224,224), batch_size=8, rotation_range=20):
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=rotation_range   
    )
    train_gen = datagen.flow_from_directory(
        folder,
        target_size=target_size,
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='binary',
        subset='training'
    )
    val_gen = datagen.flow_from_directory(
        folder,
        target_size=target_size,
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='binary',
        subset='validation'
    )
    return train_gen, val_gen

def build_model(input_shape=(224,224,1)):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Grad-CAM
def get_gradcam_heatmap(img_array, model, last_conv_layer_name=None):
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, layers.Conv2D):
                last_conv_layer_name = layer.name
                break

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap)+1e-8)
    heatmap = cv2.resize(heatmap.numpy(), (img_array.shape[2], img_array.shape[1]))
    return heatmap

def overlay_heatmap_on_image(img_pil, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    img = np.array(img_pil.convert("RGB"))
    heatmap_color = cv2.applyColorMap(np.uint8(255*heatmap), colormap)
    overlayed = cv2.addWeighted(img, 1-alpha, heatmap_color, alpha, 0)
    return Image.fromarray(overlayed)

def draw_contour_on_heatmap(img_pil, heatmap, threshold=0.5):
    mask = (heatmap >= threshold).astype(np.uint8)
    contours = measure.find_contours(mask, 0.5)
    img_draw = img_pil.convert("RGB")
    draw = ImageDraw.Draw(img_draw)
    y_scale = img_pil.height / mask.shape[0]
    x_scale = img_pil.width / mask.shape[1]
    for contour in contours:
        contour_scaled = [(c[1]*x_scale, c[0]*y_scale) for c in contour]
        draw.line(contour_scaled + [contour_scaled[0]], fill=(0,255,0), width=3)
    return img_draw

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("🔧 Settings")
target_size = st.sidebar.selectbox("Image size", [(224,224),(128,128)], index=0)
epochs = st.sidebar.number_input("Epochs", min_value=1, max_value=100, value=5)
batch_size = st.sidebar.number_input("Batch size", min_value=1, max_value=32, value=8)
heatmap_threshold = st.sidebar.slider("Threshold for contour", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
rotation_range = st.sidebar.slider("Max rotation (degrees)", min_value=0, max_value=180, value=20, step=1)

# ---------------------------
# Upload dataset & train
# ---------------------------
st.header("🏋️ Training")
st.write("Dataset: φάκελος με δύο υποφακέλους `YES` και `NO`")
uploaded_zip = st.file_uploader("Upload dataset zip", type=["zip"])
train_button = st.button("▶ Train CNN")

if train_button:
    if uploaded_zip is None:
        st.warning("Πρέπει να ανεβάσεις dataset zip πρώτα!")
    else:
        folder = extract_zip_to_folder(uploaded_zip.getvalue(), original_zip_name=uploaded_zip.name)
        st.success("Dataset εξήχθη.")

        # Data augmentation μόνο στη μνήμη
        train_gen, val_gen = create_image_generators(
            folder,
            target_size=target_size,
            batch_size=batch_size,
            rotation_range=rotation_range
        )
        st.success(f"Training samples: {train_gen.samples}, Validation samples: {val_gen.samples}")

        model = build_model(input_shape=(target_size[0], target_size[1],1))
        st.success("Model έτοιμο.")
        with st.spinner("Training σε εξέλιξη..."):
            history = model.fit(train_gen, validation_data=val_gen, epochs=epochs)
        st.success("Training ολοκληρώθηκε!")
        model.save("cnn_yesno_model.h5")
        st.download_button("💾 Κατέβασε weights (.h5)", data=open("cnn_yesno_model.h5","rb").read(), file_name="cnn_yesno_model.h5")

# ---------------------------
# Prediction panel
# ---------------------------
st.header("🔍 Πρόβλεψη & Heatmap + Contour")
uploaded_file = st.file_uploader("Upload MRI image", type=["jpg","png","jpeg"])
predict_button = st.button("▶ Predict")

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("L")
    st.subheader("MRI Image")
    st.image(pil_img, use_column_width=True)

    if predict_button:
        try:
            img_resized = pil_img.resize(target_size)
            img_array = np.expand_dims(np.array(img_resized)/255.0, axis=-1)  # (H,W,1)
            img_array = np.expand_dims(img_array, axis=0)                     # (1,H,W,1)

            if 'model' in locals():
                st.success("Χρησιμοποιείται το μοντέλο από το session.")
            else:
                st.warning("Δεν υπάρχει trained μοντέλο! Χρησιμοποιείται νέο CNN χωρίς training.")
                model = build_model(input_shape=(target_size[0], target_size[1],1))

            pred = model.predict(img_array)[0,0]
            st.success(f"Prediction: {'YES (όγκος)' if pred>0.5 else 'NO (όχι όγκος)'} ({pred:.2f})")

            heatmap = get_gradcam_heatmap(img_array, model)
            overlayed_img = overlay_heatmap_on_image(pil_img, heatmap)
            st.subheader("Heatmap (πιθανή περιοχή όγκου)")
            st.image(overlayed_img, use_column_width=True)

            contour_img = draw_contour_on_heatmap(pil_img, heatmap, threshold=heatmap_threshold)
            st.subheader("Contour γύρω από πιθανή περιοχή όγκου")
            st.image(contour_img, use_column_width=True)

        except Exception as e:
            st.error(f"Σφάλμα: {e}")
