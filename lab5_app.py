import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models
from PIL import Image
import pandas as pd

# -----------------------------
# 1. Page config
# -----------------------------
st.set_page_config(
    page_title="Image Classification with ResNet18 (CPU)",
    page_icon="🖼️",
    layout="centered"
)

st.title(":blue[Simple Image Classification] Web App")
st.caption("MOK KER QIN SD24032  \nINTRODUCTION TO ARTIFICIAL INTELLIGENCE LAB REPORT 5")
st.write("Using **PyTorch ResNet-18 (pretrained on ImageNet)** + Streamlit, running on :green[**CPU only**]")
st.divider()

# -----------------------------
# 2. CPU settings
# -----------------------------
device = torch.device("cpu")  # force CPU, no GPU needed

# -----------------------------
# 3. Utility: Load model
# -----------------------------
@st.cache_resource
def load_model():
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.to(device)
    model.eval()  # evaluation mode (no training)
    return model, weights

model, weights = load_model()

# -----------------------------
# 4. Preprocessing and labels
# -----------------------------
# Recommended transforms for ResNet18 weights:
# resize to 256, center crop 224, convert to tensor, ImageNet normalisation
preprocess = weights.transforms()
labels = weights.meta["categories"]  # 1,000 ImageNet class names

# -----------------------------
# 5. Image input (upload or webcam)
# -----------------------------
image_source = st.radio(
    "Choose an image source",
    ["Upload an image", "Take a webcam photo"],
    horizontal=True
)

if image_source == "Upload an image":
    uploaded_file = st.file_uploader(
        "Upload an image (jpg/png)",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width="stretch")
else:
    # Webcam: photo taken (left) | uploaded image (right)
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.camera_input("Take a photo")
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        with col2:
            st.image(image, caption="Uploaded Image", width="stretch")

# -----------------------------
# 6. Prediction
# -----------------------------
if uploaded_file is not None:
    # Convert image to tensor
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)  # shape: [1, 3, 224, 224]

    # Inference without gradient computation
    with torch.no_grad():
        outputs = model(input_batch)
        probabilities = F.softmax(outputs[0], dim=0)

    # Get top-5 predictions
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    # Highlight the top prediction
    st.success(
        f"Top prediction: **{labels[top5_catid[0]]}** "
        f"({top5_prob[0].item():.2%})"
    )

    st.subheader(":orange[🔍 Top-5 Predictions]")
    for i in range(top5_prob.size(0)):
        st.write(
            f"**{labels[top5_catid[i]]}** — "
            f"probability: {top5_prob[i].item():.4f}"
        )

    # Show as table
    st.write("### :violet[📋 Predictions Table]")
    df = pd.DataFrame({
        "Label": [f"{i + 1}. {labels[idx]}" for i, idx in enumerate(top5_catid)],
        "Probability": [float(p) for p in top5_prob]
    })
    st.dataframe(df, hide_index=True, width="stretch")  # hide 0-4 index

    # -----------------------------
    # 7. Bar chart of probabilities
    # -----------------------------
    st.write("### :blue[📊 Prediction Probabilities]")
    st.bar_chart(df, x="Label", y="Probability", color="#4C8BF5")

else:
    st.info("👆 Please upload an image or take a photo to start.")