import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import os
import numpy as np
from transformers import ViTModel, ViTImageProcessor  # Mudança aqui
import requests

# =========================================================
# Configuração da página (DEVE vir antes de qualquer st.xxx)
# =========================================================
st.set_page_config(page_title="Classificador Raio-X", layout="centered", initial_sidebar_state="auto", page_icon="🩻")

# =========================================================
# Dispositivo
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# Header da aplicação
# =========================================================
st.markdown(
    """
    <div style='background-color:#0E1117; padding:20px; border-radius:10px; text-align:center'>
        <h1 style='color:#C0392B'>🩻 Classificador de Raio-X <span style='color:#C0392B'> 🐾 Espécie & Região</span></h1>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# Funções de carregamento otimizadas
# =========================================================
@st.cache_resource(show_spinner=False)
def download_model():
    """Download do modelo se necessário"""
    MODEL_URL = "https://huggingface.co/pehgamarra/vit_regiao_especie/resolve/main/vit_multilabel_model.pth"
    MODEL_PATH = "vit_multilabel_model.pth"
    
    if not os.path.exists(MODEL_PATH):
        st.info("Baixando modelo...")
        try:
            r = requests.get(MODEL_URL, allow_redirects=True, timeout=60)
            r.raise_for_status()
            with open(MODEL_PATH, 'wb') as f:
                f.write(r.content)
        except Exception as e:
            st.error(f"Erro ao baixar modelo: {e}")
            st.stop()
    return MODEL_PATH

@st.cache_resource(show_spinner=False)
def load_encoders():
    """Carrega os label encoders"""
    try:
        st.info("Carregando encoders...")
        le_regiao = joblib.load("le_regiao.pkl")
        le_especie = joblib.load("le_especie.pkl")
        return le_regiao, le_especie
    except Exception as e:
        st.error(f"Erro ao carregar encoders: {e}")
        st.stop()

@st.cache_resource(show_spinner=False)
def load_model_and_processor(_le_regiao, _le_especie, model_path):
    """Carrega modelo e processador"""
    try:
        st.info("Inicializando modelo...")
        
        # Inicializar modelo
        model = ViTMultilabel(len(_le_regiao.classes_), len(_le_especie.classes_))
        
        # Carregar state_dict
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        # Carregar processador (corrigido)
        st.info("Carregando processador de imagens...")
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        
        st.success("Sistema pronto!")
        return model, processor
        
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
        st.stop()

# =========================================================
# Carregamento dos recursos
# =========================================================
try:
    # Etapa 1: Download do modelo
    model_path = download_model()
    
    # Etapa 2: Carregar encoders
    le_regiao, le_especie = load_encoders()
    
    # Etapa 3: Carregar modelo e processador
    model, feature_extractor = load_model_and_processor(le_regiao, le_especie, model_path)
    
except Exception as e:
    st.error(f"Falha crítica no sistema: {e}")
    st.stop()

# =========================================================
# Interface de upload
# =========================================================
uploaded_image = st.file_uploader("### Arraste/solte uma imagem de raio-X", type=["png","jpg","jpeg","bmp","tiff"])

if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    image.thumbnail((180, 180))
    st.image(image, use_container_width=False)

# =========================================================
# Função de predição corrigida
# =========================================================
def predict_image(image):
    try:
        # Usar o processador corrigido
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
    
    except Exception as e:
        st.error(f"Erro na predição: {e}")
        return None, None, None, None

# =========================================================
# Botão de classificação
# =========================================================
if st.button("Classificar imagem", type="primary", use_container_width=True):
    if not uploaded_image:
        st.error("Nenhuma imagem carregada!")
    else:
        with st.spinner("Classificando..."):
            result = predict_image(image)
            
            if result[0] is not None:  # Verificar se a predição foi bem-sucedida
                reg_label, reg_conf, esp_label, esp_conf = result
                
                # Limpar strings
                reg_label = str(reg_label).strip("[]'\"").capitalize()
                esp_label = str(esp_label).capitalize()

                # Mostrar resultados
                with st.chat_message("user", avatar="🐾"):
                    st.markdown(f"**Espécie:** {esp_label}")
                    st.markdown(f"<span style='font-size: 0.9em; color: gray;'>Confiança: {esp_conf:.1f}%</span>", unsafe_allow_html=True)

                with st.chat_message("user", avatar="🦴"):
                    st.markdown(f"**Região:** {reg_label}")
                    st.markdown(f"<span style='font-size: 0.9em; color: gray;'>Confiança: {reg_conf:.1f}%</span>", unsafe_allow_html=True)
            else:
                st.error("Falha na classificação. Tente novamente.")