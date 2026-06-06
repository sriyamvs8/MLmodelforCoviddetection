
# import streamlit as st
# import torch
# from PIL import Image
# from torchvision import transforms

# from login import login
# from utils import load_model

# # ---------------- SESSION ----------------

# if "logged_in" not in st.session_state:
#     st.session_state.logged_in = False

# if not st.session_state.logged_in:
#     login()
#     st.stop()

# # ---------------- PAGE ----------------

# st.set_page_config(
#     page_title="COVID AI Assistant",
#     page_icon="🩺",
#     layout="centered"
# )

# st.title("🩺 COVID Detection & Lung Analysis")

# st.write(
#     "Upload a chest X-ray image for AI-based prediction."
# )

# # ---------------- MODEL ----------------

# model = load_model()

# # ---------------- TRANSFORM ----------------

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor()
# ])

# # ---------------- FILE UPLOAD ----------------

# uploaded_file = st.file_uploader(
#     "Upload Chest X-ray",
#     type=["jpg", "jpeg", "png"]
# )

# # ---------------- PREDICTION ----------------

# if uploaded_file:

#     img = Image.open(uploaded_file).convert("RGB")

#     st.image(
#         img,
#         caption="Uploaded Chest X-ray",
#         use_container_width=True
#     )

#     img_tensor = transform(img).unsqueeze(0)

#     with torch.no_grad():
#         output = model(img_tensor)

#     probs = torch.softmax(output, dim=1)

#     pred = torch.argmax(probs, dim=1).item()

#     confidence = probs[0][pred].item() * 100

#     labels = [
#         "COVID",
#         "Lung_Opacity",
#         "Normal"
#     ]

#     prediction = labels[pred]

#     st.divider()

#     st.subheader("Diagnosis Result")

#     if prediction == "COVID":

#         st.error(
#             f"⚠️ COVID Detected ({confidence:.2f}%)"
#         )

#         st.write(
#             "The model identified lung patterns commonly associated with COVID infection."
#         )

#     elif prediction == "Lung_Opacity":

#         st.warning(
#             f"⚠️ Lung Opacity Detected ({confidence:.2f}%)"
#         )

#         st.write(
#             "The model identified opacity regions that may indicate pulmonary abnormalities."
#         )

#     else:

#         st.success(
#             f"✅ Normal Chest X-ray ({confidence:.2f}%)"
#         )

#         st.write(
#             "No major abnormal lung patterns were detected."
#         )

#     st.metric(
#         label="Confidence",
#         value=f"{confidence:.2f}%"
#     )
import streamlit as st
import torch
import cv2
import numpy as np

from PIL import Image
from torchvision import transforms

from pytorch_grad_cam.utils.image import show_cam_on_image

from login import login

from utils import (
    load_model,
    generate_gradcam,
    get_hotspot,
    region_name,
    severity_score
)

# ================= LOGIN =================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
    st.stop()

# ================= PAGE =================

st.set_page_config(
    page_title="COVID AI Assistant",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 COVID Detection + Explainable AI")

st.write(
    "Upload a Chest X-ray image for AI-powered analysis."
)

# ================= MODEL =================

model = load_model()

# ================= TRANSFORM =================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ================= FILE UPLOAD =================

uploaded_file = st.file_uploader(
    "Upload Chest X-ray",
    type=["jpg", "jpeg", "png"]
)

# ================= PREDICTION =================

if uploaded_file:

    img = Image.open(uploaded_file).convert("RGB")

    img_resized = img.resize((224, 224))

    st.image(
        img,
        caption="Uploaded Chest X-ray",
        use_container_width=True
    )

    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)

    probs = torch.softmax(output, dim=1)

    pred = torch.argmax(probs, dim=1).item()

    confidence = probs[0][pred].item() * 100

    labels = [
        "COVID",
        "Lung_Opacity",
        "Normal"
    ]

    prediction = labels[pred]

    st.divider()

    st.subheader("Diagnosis Result")

    if prediction == "COVID":

        st.error(
            f"⚠️ COVID Detected ({confidence:.2f}%)"
        )

        explanation = (
            "The model detected lung patterns commonly associated "
            "with COVID infection."
        )

    elif prediction == "Lung_Opacity":

        st.warning(
            f"⚠️ Lung Opacity Detected ({confidence:.2f}%)"
        )

        explanation = (
            "The model detected opacity regions that may indicate "
            "pulmonary abnormalities."
        )

    else:

        st.success(
            f"✅ Normal Chest X-ray ({confidence:.2f}%)"
        )

        explanation = (
            "No major abnormal lung patterns were detected."
        )

    st.write(explanation)

    st.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )

    # ================= GRAD-CAM =================

    cam = generate_gradcam(
        model,
        img_tensor
    )

    severity, severity_level = severity_score(
        cam
    )

    image_np = (
        np.array(img_resized)
        .astype(np.float32) / 255.0
    )

    overlay = show_cam_on_image(
        image_np,
        cam,
        use_rgb=True
    )

    hotspot = get_hotspot(cam)

    region = "No dominant region detected"

    zoom_crop = None

    if hotspot is not None:

        x, y, w, h = hotspot

        cv2.rectangle(
            overlay,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

        region = region_name(x, y)

        padding = 30

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)

        x2 = min(224, x + w + padding)
        y2 = min(224, y + h + padding)

        zoom_crop = (
            image_np[y1:y2, x1:x2] * 255
        ).astype(np.uint8)

    # ================= VISUALIZATION =================

    st.subheader("AI Explainability")

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            overlay,
            caption="Grad-CAM Attention Map",
            use_container_width=True
        )

    with col2:

        if zoom_crop is not None:

            st.image(
                zoom_crop,
                caption="Zoomed Attention Region",
                use_container_width=True
            )

    # ================= SUMMARY =================

    st.subheader("Clinical Assistance Summary")

    st.write(
        f"**Primary Attention Region:** {region}"
    )

    st.write(
        f"**Severity Score:** {severity}%"
    )

    st.write(
        f"**Severity Level:** {severity_level}"
    )

    if prediction == "COVID":

        st.warning(
            f"""
Primary model attention is concentrated in:

**{region}**

Severity Level: **{severity_level}**

The highlighted region influenced the
COVID prediction most strongly.
"""
        )

    elif prediction == "Lung_Opacity":

        st.warning(
            f"""
Primary model attention is concentrated in:

**{region}**

Severity Level: **{severity_level}**

Opacity-related attention detected in
this lung region.
"""
        )

    else:

        st.success(
            """
No significant abnormal attention
regions detected.
"""
        )