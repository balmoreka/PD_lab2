import sys
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, Field
from typing import List
import logging 

logging.basicConfig(level=logging.INFO)

# Crear instancia de FastAPI
app = FastAPI(
    title='API de predicción utilizando XGBoost',
    description='Servicio de API para predecir utilizando ML XGBoost',
    version='1.0.0'
)

# Cargar del modelo y preprocesadores
import os
model_name = os.getenv("MODEL")
feature_train = int(os.getenv("FEATURE_TRAIN"))
model = joblib.load('/app/models/'+model_name+'.pkl')
preprocessor = joblib.load('/app/models/preprocessor.pkl')
pca = joblib.load('/app/models/pca.pkl')

# Verificamos la salud de la API, realizamos una pequeña consulta si el servicio esta activo
# modificamos app.get con la funcion de abajo.
@app.get('/health', summary='Health check', description='Verificar el estado de la API')
async def health_check(): # summary y description son descripciones en la api
    logging.info('Health check')
    return {'status':'ok'}

# Definimos función predict.
# post
@app.post('/predict', summary='Generador de predicciones', description='Realiza predicciones basadas en las features enviadas')
async def predict(input_data:List): # funcion asincrona porque algun sistema envia info y el sistema la procesa
    logging.info('Predict request received') # similar al print
# asincrona son sistemas independientes para que no depoendan del otro sistema
    try:
        # Convertir la lista de PredictionInput a un dataframe
        df = pd.DataFrame([item for item in input_data]) # lista de diccionarios

        # aplicar preprocesador
        X = preprocessor.transform(df)
        if X.shape[1]>feature_train:
            X = pca.transform(X)
        # realizamos predicciones
        predictions = model.predict_proba(X)
        predictions = pd.DataFrame(predictions, columns=['Clase '+str(x+1) for x in range(predictions.shape[1])])
        # retornamos las predicciones utilizando JSON
        return {'predictions':predictions.to_dict('records')}
    
    except Exception as e:
        logging.error(f'Prediction failed due to {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))


