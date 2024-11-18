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

# Como se ejecuta
#EXPOSE, para que habilite el puerto 8000 para que sea utilizado por la API
EXPOSE 8000

#uvicorn, se ejecuta la instancia de fastAPI que es main.app, host es la indicación que permite la interrogar a la APi en cualquier ip 0.0.0.0, puerto definido es el 8000, que anteriormente expusimos en docker
#CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]
#CMD ["/bin/bash", "-c", "python train.py"]
#CMD ["/bin/bash","-c","uvicorn main:app --host 0.0.0.0 --port 8000"]
CMD ["/bin/bash","-c", "python train.py;uvicorn main:app --host 0.0.0.0 --port 8000"]