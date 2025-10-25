# 1. Imagem Base: Começamos com um Python 3.10 limpo
FROM python:3.10-slim

# 2. Define o diretório de trabalho dentro do container
WORKDIR /app

# 3. Instala dependências do sistema operacional
#    (OpenCV e MediaPipe precisam disso para processar imagens)
# DEPOIS (Corrigido)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copia e instala as dependências do Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copia o código do nosso pipeline para dentro do container
COPY ./pipeline /app/pipeline

# 6. Define o "ponto de entrada" (o que rodar quando o container iniciar)
#    Irá rodar o módulo 'pipeline' (o arquivo __main__.py)
ENTRYPOINT ["python", "-m", "pipeline"]