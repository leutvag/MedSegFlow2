# main.py
### MedSegFlow CNN + Heatmap + Contour + Training Plots + Confusion Matrix + Augmentation
# cd MedSegFlow/src/app
# streamlit run main.py

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import os
import zipfile
import io
import shutil
import tensorflow as tf
# from tensorflow.keras import layers, models, applications
# from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
# from tensorflow.keras.callbacks import EarlyStopping
import cv2
from skimage import measure
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="MedSegFlow CNN Analyzer", layout="centered")
st.markdown("<h1 style='text-align: center; color: rgb(200,200,125);'>🧠 MedSegFlow CNN Analyzer (YES/NO)</h1>", unsafe_allow_html=True)
st.write("Train a CNN on a YES/NO dataset (tumor / no tumor), visualize Grad-CAM heatmaps and evaluation metrics.")

# ---------------------------
# Helper functions
# ---------------------------
def extract_zip_to_folder(zip_bytes, target_folder="dataset_temp"):
    if os.path.exists(target_folder):
        shutil.rmtree(target_folder)
    os.makedirs(target_folder, exist_ok=True)
    z = zipfile.ZipFile(io.BytesIO(zip_bytes))
    z.extractall(target_folder)
    return target_folder

def create_image_generators(folder, target_size=(224,224), batch_size=8):
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=20,
        brightness_range=(0.8,1.2),
        zoom_range=0.1,
        shear_range=0.1,
        fill_mode="nearest"
    )
    train_gen = datagen.flow_from_directory(
        folder,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        shuffle=True
    )
    val_gen = datagen.flow_from_directory(
        folder,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )
    return train_gen, val_gen

def build_model(input_shape=(224,224,3)):
    base_model = applications.EfficientNetB0(
        include_top=False, input_shape=input_shape, weights='imagenet'
    )
    base_model.trainable = True
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs=base_model.input, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Grad-CAM with safe conversion to numpy
def get_gradcam_heatmap(img_array, model, last_conv_layer_name="top_conv"):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    # compute weighted map
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)

    # convert to numpy safely
    if hasattr(heatmap, "numpy"):
        heatmap_np = heatmap.numpy()
    else:
        heatmap_np = np.array(heatmap)

    heatmap_resized = cv2.resize(heatmap_np, (img_array.shape[2], img_array.shape[1]))
    # normalize 0..1
    heatmap_resized = np.maximum(heatmap_resized, 0)
    if heatmap_resized.max() > 0:
        heatmap_resized = heatmap_resized / (heatmap_resized.max() + 1e-8)
    return heatmap_resized

def overlay_heatmap_on_image(img_pil, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    img = np.array(img_pil.convert("RGB"))
    heatmap_color = cv2.applyColorMap(np.uint8(255*heatmap), colormap)
    overlayed = cv2.addWeighted(img, 1-alpha, heatmap_color, alpha, 0)
    return Image.fromarray(overlayed)

def draw_contour_on_heatmap(img_pil, heatmap, threshold=0.5):
    mask = (heatmap >= threshold).astype(np.uint8)
    if mask.sum() == 0:
        return img_pil
    contours = measure.find_contours(mask, 0.5)
    img_draw = img_pil.convert("RGB")
    draw = ImageDraw.Draw(img_draw)
    y_scale = img_pil.height / mask.shape[0]
    x_scale = img_pil.width / mask.shape[1]
    for contour in contours:
        contour_scaled = [(c[1]*x_scale, c[0]*y_scale) for c in contour]
        # draw closed polyline
        if len(contour_scaled) > 2:
            draw.line(contour_scaled + [contour_scaled[0]], fill=(0,255,0), width=3)
    return img_draw

# Augmentation that saves files to disk
def augment_dataset(input_folder, output_folder="dataset_augmented", augmentations_per_image=5, target_size=(224,224)):
    """
    input_folder: root with class subfolders (YES, NO)
    output_folder: where augmented images will be saved (same class subfolders)
    augmentations_per_image: how many augmented images per original image
    """
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    datagen = ImageDataGenerator(
        rescale=None,  # we'll save pixels 0..255
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=30,
        brightness_range=(0.8,1.2),
        zoom_range=0.15,
        shear_range=0.15,
        fill_mode="nearest"
    )

    classes = [d for d in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, d))]
    for cls in classes:
        class_in = os.path.join(input_folder, cls)
        class_out = os.path.join(output_folder, cls)
        os.makedirs(class_out, exist_ok=True)

        for fname in os.listdir(class_in):
            in_path = os.path.join(class_in, fname)
            try:
                img = load_img(in_path, target_size=target_size)
            except Exception as e:
                # skip unreadable files
                continue
            x = img_to_array(img)  # float32
            x = x.reshape((1,) + x.shape)  # (1, h, w, c)

            # save the original (resized) too
            base_name, ext = os.path.splitext(fname)
            save_path = os.path.join(class_out, f"{base_name}_orig.jpg")
            Image.fromarray(np.uint8(x[0])).save(save_path)

            i = 0
            for batch in datagen.flow(x, batch_size=1, save_to_dir=class_out,
                                      save_prefix=f"{base_name}_aug", save_format="jpg"):
                i += 1
                if i >= augmentations_per_image:
                    break
    return output_folder

# zip a folder for download
def zip_folder(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(folder_path):
            for f in files:
                full = os.path.join(root, f)
                rel = os.path.relpath(full, start=folder_path)
                zf.write(full, arcname=rel)
    return zip_path

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("🔧 Settings")
target_size = st.sidebar.selectbox("Image size", [(224,224),(128,128)], index=0)
epochs = st.sidebar.number_input("Epochs", min_value=1, max_value=100, value=5)
batch_size = st.sidebar.number_input("Batch size", min_value=1, max_value=32, value=8)
heatmap_threshold = st.sidebar.slider("Threshold for contour", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

# ---------------------------
# Dataset Upload + Augment
# ---------------------------
st.header("🏋️ Dataset Upload & Augmentation")
st.write("Upload a zip file whose root contains class subfolders (e.g. YES/ NO).")
uploaded_zip = st.file_uploader("Upload dataset zip (structure: root/YES, root/NO)", type=["zip"])

if uploaded_zip is not None:
    if st.button("🔍 Extract dataset"):
        try:
            folder = extract_zip_to_folder(uploaded_zip.getvalue(), target_folder="dataset_temp")
            st.success(f"Dataset extracted to `{folder}`")
            st.info(f"Found classes: {', '.join([d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder,d))])}")
        except Exception as e:
            st.error(f"Extract error: {e}")

    # augmentation options
    st.write("Augmentation (optional): create augmented dataset on disk")
    aug_count = st.number_input("Augmentations per image", min_value=1, max_value=50, value=5)
    if st.button("🔄 Augment dataset (save to disk)"):
        try:
            if not os.path.exists("dataset_temp"):
                st.error("No extracted dataset found. First extract the uploaded zip.")
            else:
                out_folder = augment_dataset("dataset_temp", output_folder="dataset_augmented", augmentations_per_image=int(aug_count), target_size=target_size)
                st.success(f"Augmented dataset created at `{out_folder}`")
                zip_path = "dataset_augmented.zip"
                zip_folder(out_folder, zip_path)
                with open(zip_path, "rb") as f:
                    st.download_button("💾 Download augmented dataset (zip)", data=f.read(), file_name="dataset_augmented.zip", mime="application/zip")
        except Exception as e:
            st.error(f"Augmentation error: {e}")

# ---------------------------
# Training
# ---------------------------
st.header("🏋️ Training")
st.write("Choose which folder to train on: extracted (`dataset_temp`) or augmented (`dataset_augmented`) if created.")

train_source = st.selectbox("Training source folder", options=["dataset_temp", "dataset_augmented"])
train_button = st.button("▶ Train CNN")

if train_button:
    if not os.path.exists(train_source):
        st.error(f"Training folder `{train_source}` not found. Extract or augment dataset first.")
    else:
        try:
            train_gen, val_gen = create_image_generators(train_source, target_size=target_size, batch_size=batch_size)
            st.info(f"Train samples: {train_gen.samples} | Val samples: {val_gen.samples}")

            model = build_model(input_shape=(target_size[0], target_size[1],3))
            st.info("Model built. Starting training...")

            early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

            with st.spinner("Training in progress..."):
                history = model.fit(
                    train_gen,
                    validation_data=val_gen,
                    epochs=int(epochs),
                    callbacks=[early_stop]
                )
            st.success("Training completed!")
            model.save("cnn_yesno_model.h5")
            st.download_button("💾 Download model weights (.h5)", data=open("cnn_yesno_model.h5","rb").read(), file_name="cnn_yesno_model.h5")

            # --- Plot Accuracy & Loss ---
            st.subheader("📊 Training Performance")
            fig, ax = plt.subplots(1,2, figsize=(12,5))
            ax[0].plot(history.history.get('accuracy', []), label="Train Accuracy")
            ax[0].plot(history.history.get('val_accuracy', []), label="Validation Accuracy")
            ax[0].legend(); ax[0].set_title("Accuracy")

            ax[1].plot(history.history.get('loss', []), label="Train Loss")
            ax[1].plot(history.history.get('val_loss', []), label="Validation Loss")
            ax[1].legend(); ax[1].set_title("Loss")
            st.pyplot(fig)

            # --- Confusion Matrix & Classification Report ---
            st.subheader("📌 Confusion Matrix & Classification Report")
            # ensure val_gen yields all validation samples once in order
            val_gen.reset()
            val_preds = model.predict(val_gen)
            val_preds_bin = (val_preds > 0.5).astype(int).reshape(-1)
            y_true = val_gen.classes
            cm = confusion_matrix(y_true, val_preds_bin)

            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["No Tumor","Tumor"], yticklabels=["No Tumor","Tumor"], ax=ax_cm)
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("True")
            st.pyplot(fig_cm)

            report_text = classification_report(y_true, val_preds_bin, target_names=["No Tumor","Tumor"])
            st.text(report_text)

        except Exception as e:
            st.error(f"Training error: {e}")

# ---------------------------
# Prediction panel
# ---------------------------
st.header("🔍 Prediction & Heatmap + Contour")
uploaded_file_img = st.file_uploader("Upload MRI image for prediction", type=["jpg","png","jpeg"])
predict_button = st.button("▶ Predict Image")

if uploaded_file_img is not None:
    pil_img = Image.open(uploaded_file_img).convert("RGB")
    st.subheader("MRI Image")
    st.image(pil_img, use_column_width=True)

    if predict_button:
        try:
            img_resized = pil_img.resize(target_size)
            img_array = np.expand_dims(np.array(img_resized)/255.0, axis=0)

            if not os.path.exists("cnn_yesno_model.h5"):
                st.error("No trained model found. Train a model first or upload cnn_yesno_model.h5 to the app folder.")
            else:
                model = tf.keras.models.load_model("cnn_yesno_model.h5", compile=False)
                pred = float(model.predict(img_array)[0,0])
                st.success(f"Prediction: {'YES (Tumor)' if pred>0.5 else 'NO (No Tumor)'} ({pred:.3f})")

                heatmap = get_gradcam_heatmap(img_array, model)
                overlayed_img = overlay_heatmap_on_image(pil_img, heatmap)
                st.subheader("Grad-CAM Heatmap (Potential Tumor Area)")
                st.image(overlayed_img, use_column_width=True)

                contour_img = draw_contour_on_heatmap(pil_img, heatmap, threshold=heatmap_threshold)
                st.subheader("Contour around potential tumor area")
                st.image(contour_img, use_column_width=True)

                # allow download of overlay and contour
                buf = io.BytesIO()
                overlayed_img.save(buf, format="PNG"); buf.seek(0)
                st.download_button("💾 Download overlay image", data=buf.getvalue(), file_name="overlay.png", mime="image/png")

                buf2 = io.BytesIO()
                contour_img.save(buf2, format="PNG"); buf2.seek(0)
                st.download_button("💾 Download contour image", data=buf2.getvalue(), file_name="contour.png", mime="image/png")

        except Exception as e:
            st.error(f"Prediction error: {e}")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown("""
**About this app:**  
This Streamlit app trains a CNN to classify MRI brain images into **YES (tumor)** or **NO (no tumor)**.  
Features included:
- Dataset extraction from uploaded zip (root must contain class subfolders, e.g. `YES` and `NO`).
- Optional on-disk augmentation (creates `dataset_augmented` and offers download as zip).
- Training with EfficientNetB0 backbone, early stopping.
- Training plots (accuracy & loss), confusion matrix and classification report.
- Grad-CAM heatmap and contour overlay on uploaded test images.
- Download buttons for model weights & generated images.
""")


