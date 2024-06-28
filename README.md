# IA de Reconhecimento Facial

Este é um projeto de reconhecimento facial baseado em tecnologias de deep learning e machine learning, utilizando OpenCV e face_recognition para detecção e reconhecimento de faces.

## Começando

Este projeto é um ponto de partida para um sistema de reconhecimento facial. Siga as instruções abaixo para configurar o projeto em sua máquina local.

### Pré-requisitos

Antes de iniciar, certifique-se de ter o seguinte instalado:
- Python 3.x
- OpenCV
- face_recognition
- dlib

### Instalação

1. Clone o repositório para a sua máquina local usando:
   ```bash
   git clone https://github.com/seudominio/ia-reconhecimento-facial.git
2. Navegue até o diretório do projeto:
   ```bash
   cd ia-reconhecimento-facial
3. Crie um ambiente virtual (opcional, mas recomendado):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Para Linux e macOS
   .\.venv\Scripts\activate   # Para Windows
4. Instale as dependências do projeto executando:
   ```bash
    pip install -r requirements.txt


### Execução
1. Para capturar imagens com a webcam e treinar o modelo, execute:
   ```bash
   python face_capture_webcam.py
   
2. Para treinar os reconhecedores, execute:
   ```bash
   python train_recognizers.py

3. Para executar o reconhecimento facial usando deep learning, execute:
   ```bash
   python recognition_deeplearning_webcam.py

4. Para executar o reconhecimento facial com base nos reconhecedores treinados, execute:
   ```bash
   python recognition_webcam.py

### Arquitetura do Projeto
- face_capture_webcam.py: Captura imagens usando a webcam e salva em um diretório de treinamento.
- train_recognizers.py: Treina os modelos de reconhecimento facial (Eigenfaces, Fisherfaces, LBPH).
- recognition_deeplearning_webcam.py: Reconhecimento facial em tempo real usando deep learning (face_recognition).
- recognition_webcam.py: Reconhecimento facial em tempo real usando modelos treinados (Eigenfaces, Fisherfaces, LBPH).
- encodingfaces.py: Codifica rostos a partir de imagens para reconhecimento facial.
