import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import os
import numpy as np
from transformers import ViTModel, ViTFeatureExtractor

# =========================================================
# Modelo ViT Multilabel
# =========================================================

class ViTMultilabel(nn.Module):
    def __init__(self, num_regioes, num_especies):
        super(ViTMultilabel, self).__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        hidden_size = self.vit.config.hidden_size
        self.classifier_regiao = nn.Linear(hidden_size, num_regioes)
        self.classifier_especie = nn.Linear(hidden_size, num_especies)

    def forward(self, x):
        batch_size, n_imgs, C, H, W = x.shape
        x = x.view(-1, C, H, W)
        outputs = self.vit(pixel_values=x)
        pooled = outputs.pooler_output.view(batch_size, n_imgs, -1)
        agg = pooled.mean(dim=1)
        reg_logits = self.classifier_regiao(agg)
        esp_logits = self.classifier_especie(agg)
        return reg_logits, esp_logits

# =========================================================
# Configuração da página
# =========================================================

st.set_page_config(page_title="Classificador Raio-X", layout="centered", initial_sidebar_state="auto", page_icon="🩻")
st.markdown(
    """
    <div style='background-color:#0E1117; padding:20px; border-radius:10px; text-align:center'>
        <h1 style='color:#C0392B'>🩻 Classificador de Raio-X <span style='color:#C0392B'> 🐾 Espécie & Região</span></h1>
    </div>
    """,
    unsafe_allow_html=True
)
# =========================================================
# Dispositivo
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# Carregar recursos automaticamente
# =========================================================

@st.cache_resource(show_spinner=True)
def load_assets():
    # encoders
    le_regiao = joblib.load("le_regiao.pkl")
    le_especie = joblib.load("le_especie.pkl")
    # modelo
    model = ViTMultilabel(len(le_regiao.classes_), len(le_especie.classes_))
    model.load_state_dict(torch.load("vit_multilabel_model.pth", map_location=device))
    model.to(device)
    model.eval()
    # feature extractor
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
    return model, feature_extractor, le_regiao, le_especie

model, feature_extractor, le_regiao, le_especie = load_assets()

# =========================================================
# Upload de imagem
# =========================================================

uploaded_image = st.file_uploader("### Arraste/solte uma imagem de raio-X", type=["png","jpg","jpeg","bmp","tiff"])
if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    image.thumbnail((360, 360))
    st.image(image, caption="Miniatura da imagem", use_container_width=False)

# =========================================================
# Predição
# =========================================================

def predict_image(image):
    inputs = feature_extractor(images=image, return_tensors="pt")
    pixel_values = inputs['pixel_values'].unsqueeze(1).to(device)  # 1 exame, 1 imagem

    with torch.no_grad():
        reg_logits, esp_logits = model(pixel_values)
        reg_probs = F.softmax(reg_logits, dim=1).cpu().numpy()[0]
        esp_probs = F.softmax(esp_logits, dim=1).cpu().numpy()[0]

    reg_idx = int(np.argmax(reg_probs))
    esp_idx = int(np.argmax(esp_probs))
    reg_label = le_regiao.inverse_transform([reg_idx])[0]
    esp_label = le_especie.inverse_transform([esp_idx])[0]
    reg_conf = reg_probs[reg_idx] * 100
    esp_conf = esp_probs[esp_idx] * 100

    return reg_label, reg_conf, esp_label, esp_conf

# =========================================================
# Botão para classificar
# =========================================================

if st.button("Classificar imagem", type="primary", use_container_width=True):
    if not uploaded_image:
        st.error("Nenhuma imagem carregada!")
    else:
        with st.spinner("Classificando..."):
            reg_label, reg_conf, esp_label, esp_conf = predict_image(image)
            st.markdown("### Resultado")
            st.write(f"**Espécie:** {esp_label} — Confiança: {esp_conf:.1f}%")
            st.write(f"**Região:** {reg_label} — Confiança: {reg_conf:.1f}%")