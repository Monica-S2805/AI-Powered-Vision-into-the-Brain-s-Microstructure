import os

import numpy as np
import streamlit as st
from PIL import Image

try:
    from tensorflow.keras.models import load_model
except ImportError:
    load_model = None  # type: ignore[misc, assignment]

models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
model_file_path = os.path.join(models_dir, "BrainTumor10EpochsCategorical.h5")


def get_className(classNo):
    if classNo == 0:
        return "No Brain Tumor"
    elif classNo == 1:
        return "Yes Brain Tumor"
    return "Unknown"


def get_prediction(model, image):
    image = image.convert("RGB")
    image = image.resize((64, 64))
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result = model.predict(input_img)
    class_index = np.argmax(result)
    return class_index


def main():
    st.set_page_config(page_title="Brain Tumor", layout="wide", page_icon="🧠")
    st.title("Brain Tumor Prediction")
    st.text("Upload an MRI scan image to check for brain tumor")

    if load_model is None:
        st.error(
            "**TensorFlow is not available.** Official builds do not support Python 3.14 yet. "
            "Install **Python 3.11 or 3.12** (e.g. `brew install python@3.12`), then:\n\n"
            "`python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`\n\n"
            "Restart Streamlit from this folder."
        )
        st.stop()

    if not os.path.exists(model_file_path):
        st.error(f"Model file not found: `{model_file_path}`")
        st.stop()

    if "brain_tumor_model" not in st.session_state:
        st.session_state["brain_tumor_model"] = load_model(model_file_path)
    model = st.session_state["brain_tumor_model"]

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        prediction = get_prediction(model, image)
        class_name = get_className(prediction)

        st.image(image, caption="Uploaded Image", use_column_width=True)

        st.write(f"##### Prediction: {class_name}")

        if class_name == "Yes Brain Tumor":
            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)

            st.markdown("## Brain Tumor Information")
            st.write("**📌 Common Name:** Brain Tumor")
            st.write(
                "**🌐 General Overview:** A brain tumor is an abnormal growth of cells within the brain."
            )

            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)

            st.markdown("### Symptoms")
            st.write("🧠 Symptoms of Brain Tumor include:")
            st.markdown("- Headaches.")
            st.markdown("- Seizures.")
            st.markdown("- Cognitive and memory problems.")

            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)

            st.markdown("### Diagnosis")
            st.write("🩺 Diagnosis involves:")
            st.markdown("- Imaging tests such as MRI and CT scans.")
            st.markdown("- Biopsy for confirming the tumor type.")

            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)

            st.markdown("### Treatment")
            st.write("💼 Treatment options depend on:")
            st.markdown("- Tumor type, size, and location.")
            st.markdown(
                "- They may include surgery, radiation therapy, chemotherapy, and targeted therapy."
            )

            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)

            st.markdown("### Prognosis")
            st.write("⏳ Prognosis varies based on:")
            st.markdown("- Tumor type and stage.")
            st.markdown("- Treatment response.")

            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)

            st.markdown("### Living with a Brain Tumor")
            st.write("🤝 Managing life with a brain tumor:")
            st.markdown("- Regular medical checkups.")
            st.markdown("- Support groups and counseling.")

            st.markdown('<hr style="border: 1px solid #ddd;">', unsafe_allow_html=True)

            st.markdown("### More Information")
            st.write(
                "📚 For more detailed information about Brain Tumors, you can visit reputable medical websites or consult with a healthcare professional."
            )


if __name__ == "__main__":
    main()
