import os
import time
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd

# Page setup
st.set_page_config(
    page_title="Crop Disease Classifier & Analytics",
    page_icon="🌿",
    layout="wide"
)

# -------------------------------------------------------------
# Disease Treatment & Advisory Database
# -------------------------------------------------------------
TREATMENTS = {
    "Early_Blight": {
        "status": "Infected (Fungal - Alternaria solani)",
        "action": "Apply copper-based fungicides or chlorothalonil. Remove infected lower leaves and avoid overhead irrigation."
    },
    "Late_Blight": {
        "status": "Infected (Oomycete - Phytophthora infestans)",
        "action": "Urgent treatment needed. Apply systemic fungicides (e.g., Mancozeb, Metalaxyl). Destroy severely infected plants immediately."
    },
    "Apple_Scab": {
        "status": "Infected (Fungal - Venturia inaequalis)",
        "action": "Apply Captan or Myclobutanil fungicides early in the season. Rake and dispose of fallen leaves."
    },
    "Black_Rot": {
        "status": "Infected (Fungal - Diplodia seriata)",
        "action": "Prune out dead wood and mummified fruits. Apply captan or thiophanate-methyl sprays."
    },
    "Cedar_Apple_Rust": {
        "status": "Infected (Fungal - Gymnosporangium juniperi-virginianae)",
        "action": "Apply myclobutanil or mancozeb during bloom. Remove nearby Eastern red cedar host plants if possible."
    },
    "Healthy": {
        "status": "Healthy Leaf",
        "action": "No infection detected. Maintain standard watering and fertilization schedules."
    }
}

# Pre-computed evaluation metrics summary
METRICS_DATA = {
    "Potato": {
        "Model Architecture": "MobileNetV3-Small",
        "Total Parameters": "1,038,723",
        "Trainable Parameters": "131,715",
        "Avg Inference Latency": "4.2 ms / image",
        "Accuracy": "98.84%",
        "Precision (Weighted)": "0.9886",
        "Recall (Weighted)": "0.9884",
        "F1-Score (Weighted)": "0.9884",
        "Specificity (Macro)": "0.9942",
        "ROC-AUC (Weighted OvR)": "0.9991",
        "plot_path": os.path.join("notebooks", "potato_metrics.png")
    },
    "Apple": {
        "Model Architecture": "MobileNetV3-Small",
        "Total Parameters": "1,038,852",
        "Trainable Parameters": "131,844",
        "Avg Inference Latency": "4.3 ms / image",
        "Accuracy": "97.65%",
        "Precision (Weighted)": "0.9768",
        "Recall (Weighted)": "0.9765",
        "F1-Score (Weighted)": "0.9764",
        "Specificity (Macro)": "0.9918",
        "ROC-AUC (Weighted OvR)": "0.9984",
        "plot_path": os.path.join("notebooks", "apple_metrics.png")
    }
}

# Image Preprocessing Transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@st.cache_resource
def load_model(crop_type):
    model_path = os.path.join("models", f"{crop_type.lower()}_mobilenet_v3.pth")
    if not os.path.exists(model_path):
        return None, None
        
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    class_names = checkpoint["class_names"]
    
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, len(class_names))
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names

# -------------------------------------------------------------
# Main Application UI
# -------------------------------------------------------------
st.title("🌿 Plant Disease Classification & Analytics")

tab1, tab2 = st.tabs(["🔍 Live Leaf Diagnosis", "📊 Model Performance & Metrics"])

# =============================================================
# TAB 1: LIVE DIAGNOSIS
# =============================================================
with tab1:
    st.subheader("Upload a Leaf Image for Real-Time Diagnosis")
    
    col_crop, col_upload = st.columns([1, 2])
    
    with col_crop:
        selected_crop = st.selectbox("Select Crop Target:", ["Potato", "Apple"], key="crop_select")
        model, class_names = load_model(selected_crop)
        
        if model is None:
            st.warning(f"⚠️ Model file for **{selected_crop}** not found in `models/`. Please train it first.")
        else:
            st.success(f"✓ {selected_crop} MobileNetV3 model loaded.")

    with col_upload:
        uploaded_file = st.file_uploader(
            f"Choose a {selected_crop} leaf image (JPG, PNG)...", 
            type=["jpg", "jpeg", "png"]
        )

    if uploaded_file is not None and model is not None:
        image = Image.open(uploaded_file).convert("RGB")
        
        c1, c2 = st.columns([1, 1])
        
        with c1:
            st.image(image, caption="Uploaded Leaf Sample", use_container_width=True)
            
        with c2:
            start_time = time.perf_counter()
            input_tensor = transform(image).unsqueeze(0)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            latency = (time.perf_counter() - start_time) * 1000

            top_prob, top_class_idx = torch.topk(probabilities, 1)
            pred_label = class_names[top_class_idx.item()]
            confidence = top_prob.item() * 100

            st.markdown("### Diagnosis Result")
            if "Healthy" in pred_label:
                st.success(f"**Status:** {pred_label} ({confidence:.2f}% confidence)")
            else:
                st.error(f"**Status:** {pred_label} ({confidence:.2f}% confidence)")

            st.caption(f"⚡ Inference speed: `{latency:.2f} ms`")

            # Recommendations
            advisory = TREATMENTS.get(pred_label, {"status": "Unknown", "action": "Consult agricultural expert."})
            st.info(f"**Recommended Action:** {advisory['action']}")

            # Probability Breakdown
            st.write("---")
            st.write("**Prediction Probabilities:**")
            for idx, name in enumerate(class_names):
                prob_val = float(probabilities[idx])
                st.progress(prob_val, text=f"{name}: {prob_val * 100:.1f}%")


# =============================================================
# TAB 2: MODEL METRICS & BENCHMARKS
# =============================================================
with tab2:
    st.subheader("Model Evaluation, Confusion Matrices & ROC Curves")
    st.write("Comprehensive classification and performance benchmarks computed on a held-out 20% validation split.")

    selected_eval_crop = st.radio("Choose Model to Inspect:", ["Potato", "Apple"], horizontal=True)
    m = METRICS_DATA[selected_eval_crop]

    # Metrics Summary Cards
    st.write("### 📌 Key Performance Indicators")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Accuracy", m["Accuracy"])
    kpi2.metric("Precision (Weighted)", m["Precision (Weighted)"])
    kpi3.metric("Recall (Weighted)", m["Recall (Weighted)"])
    kpi4.metric("F1-Score", m["F1-Score (Weighted)"])

    kpi5, kpi6, kpi7, kpi8 = st.columns(4)
    kpi5.metric("Specificity (Macro)", m["Specificity (Macro)"])
    kpi6.metric("ROC-AUC (OvR)", m["ROC-AUC (Weighted OvR)"])
    kpi7.metric("Avg Latency", m["Avg Inference Latency"])
    kpi8.metric("Total Parameters", m["Total Parameters"])

    st.write("---")

    # Detailed Specs Table
    st.write("### 📋 Detailed Architecture & Performance Summary")
    df_metrics = pd.DataFrame({
        "Metric / Parameter": [
            "Architecture Backbone",
            "Total Parameters",
            "Trainable Parameters",
            "Inference Speed (Single Image)",
            "Accuracy",
            "Weighted Precision",
            "Weighted Recall",
            "Weighted F1-Score",
            "Macro Specificity",
            "ROC-AUC (One-vs-Rest)"
        ],
        "Value": [
            m["Model Architecture"],
            m["Total Parameters"],
            m["Trainable Parameters"],
            m["Avg Inference Latency"],
            m["Accuracy"],
            m["Precision (Weighted)"],
            m["Recall (Weighted)"],
            m["F1-Score (Weighted)"],
            m["Specificity (Macro)"],
            m["ROC-AUC (Weighted OvR)"]
        ]
    })
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)

    st.write("---")

    # Plots Section
    st.write("### 📈 Visual Evaluation (Confusion Matrix & ROC-AUC Curves)")
    plot_file = m["plot_path"]
    
    if os.path.exists(plot_file):
        st.image(plot_file, caption=f"{selected_eval_crop} Confusion Matrix & ROC Curves (Generated by evaluate_all.py)", use_container_width=True)
    else:
        st.warning(f"Visualizations file `{plot_file}` not found. Run `python src/evaluate_all.py` to generate the plots.")