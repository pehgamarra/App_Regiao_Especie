from setuptools import setup
import sys

# Configuração para cx_Freeze
if sys.platform == "win32":
    from cx_Freeze import setup, Executable
    
    # Dependências que precisam ser incluídas
    packages = [
        "tkinter", "PIL", "numpy", "torch", "transformers", 
        "joblib", "pathlib", "os", "sys"
    ]
    
    # Arquivos adicionais que devem ser copiados
    include_files = [
        # Adicione seus arquivos de modelo
        ("vit_multilabel_model.pth", "vit_multilabel_model.pth"),
        ("le_regiao.pkl", "le_regiao.pkl"),
        ("le_especie.pkl", "le_especie.pkl"),
    ]
    
    # Opções de build
    build_exe_options = {
        "packages": packages,
        "include_files": include_files,
        "excludes": ["matplotlib", "scipy", "tensorflow"],  # Excluir pacotes desnecessários
        "optimize": 2,
    }
    
    # Definir o executável
    executables = [
        Executable(
            script="xray_classifier.py",
            base="Win32GUI",  # Remove console window
            target_name="ClassificadorRaioX.exe",
            icon=None  # Adicione um ícone se quiser
        )
    ]
    
    setup(
        name="Classificador de Raio-X",
        version="1.0",
        description="Aplicação para classificar raios-X por espécie e região",
        options={"build_exe": build_exe_options},
        executables=executables
    )

# Comando: pyinstaller --onefile --windowed --name="ClassificadorRaioX" xray_classifier.py