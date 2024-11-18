# Base, son librerias ya creadas (imagenes), ver en el dockerhub python
FROM python:3.12-slim

# Determinamos nuestro directorio de trabajo
WORKDIR /app

# Dependencias y librerias. Hacer un instalador de dependencias y librerias a utilizar
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && apt-get clean
    
# Agregamos nuestros archivos # todos los archivos de la carpeta agregarlos a /app
COPY . /app


# Instalar requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Como se ejecuta. Al darle docker run que ejecutara el contenedor
CMD ["/bin/bash", "-c", "python train.py; python detect_changes.py"]
# ENTRYPOINT ["python", "detect_changes.py"]
