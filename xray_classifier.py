import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import torch
import joblib
import numpy as np
import torch.nn as nn   
from pathlib import Path
import os
from transformers import ViTModel, ViTFeatureExtractor

class ViTMultilabel(nn.Module):
    def __init__(self, num_regioes, num_especies):
        super(ViTMultilabel, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        hidden_size = self.vit.config.hidden_size
        self.classifier_regiao = nn.Linear(hidden_size, num_regioes)
        self.classifier_especie = nn.Linear(hidden_size, num_especies)
    
    def forward(self, x):
        batch_size, n_imgs, C, H, W = x.shape
        x = x.view(-1, C, H, W)
        outputs = self.vit(pixel_values=x)
        pooled_output = outputs.pooler_output
        pooled_output = pooled_output.view(batch_size, n_imgs, -1)
        agg_output = pooled_output.mean(dim=1)
        regiao_logits = self.classifier_regiao(agg_output)
        especie_logits = self.classifier_especie(agg_output)
        return regiao_logits, especie_logits
        
class XRayClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Classificador de Raio-X - Espécie e Região")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')
        
        # Variáveis para armazenar o modelo e encoders
        self.model = None
        self.le_regiao = None
        self.le_especie = None
        self.feature_extractor = None
        self.device = None
        self.model_loaded = False
        
        # Configurar estilo
        self.setup_styles()
        
        # Criar interface
        self.create_widgets()
        
        # Tentar carregar modelo automaticamente
        self.load_model_auto()
    
    def setup_styles(self):
        """Configurar estilos da interface"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configurar cores personalizadas
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), background='#f0f0f0')
        style.configure('Result.TLabel', font=('Arial', 12), background='white', 
                       relief='solid', borderwidth=1, padding=10)
    
    def create_widgets(self):
        """Criar widgets da interface"""
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Título
        title_label = ttk.Label(main_frame, text="Classificador de Raio-X", 
                               style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Frame para seleção de modelo
        model_frame = ttk.LabelFrame(main_frame, text="Modelo e Encoders", padding="10")
        model_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.model_status_label = ttk.Label(model_frame, text="Modelo não carregado")
        self.model_status_label.grid(row=0, column=0, sticky=tk.W)
        
        load_model_btn = ttk.Button(model_frame, text="Carregar Arquivos do Modelo", 
                                   command=self.load_model_files)
        load_model_btn.grid(row=0, column=1, padx=(10, 0))
        
        # Labels de status dos componentes
        self.encoder_status_label = ttk.Label(model_frame, text="Encoders: Não carregados")
        self.encoder_status_label.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))
        
        # Frame para seleção de arquivo
        file_frame = ttk.LabelFrame(main_frame, text="Selecionar Imagem", padding="10")
        file_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.file_path_var = tk.StringVar()
        self.file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, 
                                   width=60, state='readonly')
        self.file_entry.grid(row=0, column=0, padx=(0, 10))
        
        select_file_btn = ttk.Button(file_frame, text="Selecionar Arquivo", 
                                    command=self.select_file)
        select_file_btn.grid(row=0, column=1)
        
        # Botão de classificar
        self.classify_btn = ttk.Button(main_frame, text="Classificar Raio-X", 
                                      command=self.classify_image, state='disabled')
        self.classify_btn.grid(row=3, column=1, pady=20)
        
        # Frame para exibição da imagem
        image_frame = ttk.LabelFrame(main_frame, text="Imagem", padding="10")
        image_frame.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        self.image_label = ttk.Label(image_frame, text="Nenhuma imagem selecionada")
        self.image_label.grid(row=0, column=0)
        
        # Frame para resultados
        result_frame = ttk.LabelFrame(main_frame, text="Resultados", padding="10")
        result_frame.grid(row=4, column=1, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Labels para resultados
        ttk.Label(result_frame, text="Espécie:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.species_result = ttk.Label(result_frame, text="-", style='Result.TLabel')
        self.species_result.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=5)
        
        ttk.Label(result_frame, text="Região:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.region_result = ttk.Label(result_frame, text="-", style='Result.TLabel')
        self.region_result.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=5)
        
        ttk.Label(result_frame, text="Conf. Espécie:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.species_confidence_result = ttk.Label(result_frame, text="-", style='Result.TLabel')
        self.species_confidence_result.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=5)
        
        ttk.Label(result_frame, text="Conf. Região:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.region_confidence_result = ttk.Label(result_frame, text="-", style='Result.TLabel')
        self.region_confidence_result.grid(row=3, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=5)
        
        # Configurar redimensionamento
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(4, weight=1)
        result_frame.columnconfigure(1, weight=1)
    
    def load_model_auto(self):
        """Tentar carregar modelo automaticamente"""
        try:
            model_path = 'vit_multilabel_model.pth'
            regiao_encoder_path = 'le_regiao.pkl'
            especie_encoder_path = 'le_especie.pkl'
            
            if all(os.path.exists(path) for path in [model_path, regiao_encoder_path, especie_encoder_path]):
                self.load_model_components(model_path, regiao_encoder_path, especie_encoder_path)
            else:
                missing_files = []
                if not os.path.exists(model_path):
                    missing_files.append('vit_multilabel_model.pth')
                if not os.path.exists(regiao_encoder_path):
                    missing_files.append('le_regiao.pkl')
                if not os.path.exists(especie_encoder_path):
                    missing_files.append('le_especie.pkl')
                
                self.model_status_label.config(text=f"Arquivos não encontrados: {', '.join(missing_files)}")
        except Exception as e:
            self.model_status_label.config(text=f"Erro ao carregar automaticamente: {str(e)}")
    
    def load_model_files(self):
        """Carregar arquivos do modelo manualmente"""
        # Selecionar modelo PyTorch
        model_path = filedialog.askopenfilename(
            title="Selecionar modelo PyTorch (.pth)",
            filetypes=[("Modelo PyTorch", "*.pth"), ("Todos os arquivos", "*.*")]
        )
        
        if not model_path:
            return
        
        # Selecionar encoder de região
        regiao_encoder_path = filedialog.askopenfilename(
            title="Selecionar encoder de região (le_regiao.pkl)",
            filetypes=[("Pickle files", "*.pkl"), ("Todos os arquivos", "*.*")]
        )
        
        if not regiao_encoder_path:
            return
        
        # Selecionar encoder de espécie
        especie_encoder_path = filedialog.askopenfilename(
            title="Selecionar encoder de espécie (le_especie.pkl)",
            filetypes=[("Pickle files", "*.pkl"), ("Todos os arquivos", "*.*")]
        )
        
        if not especie_encoder_path:
            return
        
        # Carregar os componentes
        self.load_model_components(model_path, regiao_encoder_path, especie_encoder_path)
    
    def load_model_components(self, model_path, regiao_encoder_path, especie_encoder_path):
        """Carregar modelo e encoders"""
        try:
            # Carregar encoders
            self.le_regiao = joblib.load(regiao_encoder_path)
            self.le_especie = joblib.load(especie_encoder_path)
            
            # Configurar device
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Instanciar modelo
            self.model = ViTMultilabel(
                num_regioes=len(self.le_regiao.classes_),
                num_especies=len(self.le_especie.classes_)
            )
            
            # Carregar pesos do modelo
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            self.model = self.model.to(self.device)
            
            # Carregar feature extractor
            self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
            
            self.model_loaded = True
            
            # Atualizar interface
            device_name = "GPU" if self.device.type == 'cuda' else "CPU"
            self.model_status_label.config(text=f"Modelo carregado - Device: {device_name}")
            self.encoder_status_label.config(
                text=f"Encoders: Região ({len(self.le_regiao.classes_)} classes), "
                     f"Espécie ({len(self.le_especie.classes_)} classes)"
            )
            
            messagebox.showinfo("Sucesso", "Modelo e encoders carregados com sucesso!")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar modelo:\n{str(e)}")
            self.model_loaded = False
    
    def select_file(self):
        """Selecionar arquivo de imagem"""
        filetypes = [
            ("Imagens", "*.png *.jpg *.jpeg *.bmp *.tiff *.dicom"),
            ("Todos os arquivos", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Selecionar imagem de raio-X",
            filetypes=filetypes
        )
        
        if filename:
            self.file_path_var.set(filename)
            self.display_image(filename)
            
            # Habilitar botão de classificar se modelo estiver carregado
            if self.model_loaded:
                self.classify_btn.config(state='normal')
    
    def display_image(self, image_path):
        """Exibir imagem selecionada"""
        try:
            # Carregar e redimensionar imagem
            image = Image.open(image_path)
            
            # Redimensionar mantendo proporção
            max_size = (300, 300)
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Converter para PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Atualizar label
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # Manter referência
            
        except Exception as e:
            self.image_label.config(text=f"Erro ao carregar imagem: {str(e)}")
    
    def classify_image(self):
        """Classificar imagem selecionada"""
        if not self.model_loaded:
            messagebox.showerror("Erro", "Modelo não carregado!")
            return
        
        image_path = self.file_path_var.get()
        if not image_path:
            messagebox.showerror("Erro", "Nenhuma imagem selecionada!")
            return
        
        try:
            # Carregar e preparar a imagem
            image = Image.open(image_path).convert('RGB')
            
            # Usar o feature extractor do ViT
            inputs = self.feature_extractor(images=image, return_tensors="pt")
            pixel_values = inputs['pixel_values']  # (1, 3, 224, 224)
            
            # Simular batch com 1 exame e 1 imagem
            pixel_values = pixel_values.unsqueeze(1)  # (1, 1, 3, 224, 224)
            pixel_values = pixel_values.to(self.device)
            
            # Fazer predição
            with torch.no_grad():
                regiao_logits, especie_logits = self.model(pixel_values)
                
                # Calcular probabilidades usando softmax
                regiao_probs = torch.softmax(regiao_logits, dim=1)
                especie_probs = torch.softmax(especie_logits, dim=1)
                
                # Obter predições
                regiao_pred = torch.argmax(regiao_logits, dim=1).cpu().item()
                especie_pred = torch.argmax(especie_logits, dim=1).cpu().item()
                
                # Obter confidências
                regiao_confidence = regiao_probs[0, regiao_pred].cpu().item() * 100
                especie_confidence = especie_probs[0, especie_pred].cpu().item() * 100
            
            # Converter para labels legíveis
            regiao_label = self.le_regiao.inverse_transform([regiao_pred])[0]
            especie_label = self.le_especie.inverse_transform([especie_pred])[0]
            
            # Atualizar interface
            self.region_result.config(text=regiao_label)
            self.species_result.config(text=especie_label)
            self.region_confidence_result.config(text=f"{regiao_confidence:.1f}%")
            self.species_confidence_result.config(text=f"{especie_confidence:.1f}%")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro na classificação:\n{str(e)}")
    
    def run(self):
        """Executar aplicação"""
        self.root.mainloop()

def main():
    """Função principal"""
    root = tk.Tk()
    app = XRayClassifierApp(root)
    app.run()

if __name__ == "__main__":
    main()