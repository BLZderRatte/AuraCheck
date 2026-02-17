import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from keras.models import load_model

# ───────────────────────────────────────────────
#  Seite konfigurieren
# ───────────────────────────────────────────────
st.set_page_config(
    page_title="Bild-Klassifikator",
    page_icon="🖼️",
    layout="centered"
)

st.title("Bild-Klassifikator mit eigenem Modell")
st.markdown("Lade ein Bild hoch – das Modell sagt dir, was es erkennt.")

# ───────────────────────────────────────────────
# Model & Labels laden (wird nur einmal ausgeführt)
# ───────────────────────────────────────────────
@st.cache_resource
def load_my_model():
    try:
        model = load_model("keras_Model.h5", compile=False)
        class_names = open("labels.txt", "r").readlines()
        class_names = [line.strip() for line in class_names]  # sauberer
        return model, class_names
    except Exception as e:
        st.error(f"Modell oder labels.txt konnte nicht geladen werden:\n{e}")
        st.stop()

model, class_names = load_my_model()

# ───────────────────────────────────────────────
# Upload-Bereich
# ───────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Wähle ein Bild …",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=False
)

if uploaded_file is not None:
    # Bild anzeigen
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    # ───────────────────────────────────────────────
    # Preprocessing (genau wie in deinem Original-Code)
    # ───────────────────────────────────────────────
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # in numpy array umwandeln & normalisieren
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Batch-Dimension hinzufügen → Shape (1, 224, 224, 3)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # ───────────────────────────────────────────────
    # Vorhersage
    # ───────────────────────────────────────────────
    with st.spinner("Analysiere Bild …"):
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Teachable Machine Labels haben meist "0 ", "1 " etc. am Anfang
        if class_name[0].isdigit() and class_name[1] == " ":
            class_name = class_name[2:].strip()

    # ───────────────────────────────────────────────
    # Ergebnis schön darstellen
    # ───────────────────────────────────────────────
    st.success("Analyse abgeschlossen!")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**Erkannt:**  {class_name}")
    with col2:
        st.metric("Konfidenz", f"{confidence_score:.1%}")

    # Balken für Spaß und Übersicht
    st.progress(float(confidence_score))
    
    # Alle Wahrscheinlichkeiten (optional)
    if st.checkbox("Alle Klassen + Wahrscheinlichkeiten anzeigen"):
        for i, prob in enumerate(prediction[0]):
            label = class_names[i]
            if label[0].isdigit() and label[1] == " ":
                label = label[2:].strip()
            st.write(f"{label:.<30} {prob:.3%}")
