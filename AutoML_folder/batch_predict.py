# cargar de parametros y seleccion de modelo

def load_params():
    import os
    model_name = os.getenv("MODEL")
    feature_train = int(os.getenv("FEATURE_TRAIN"))
    return model_name, feature_train
def load_model(model_name):
    import joblib
    model = joblib.load('/app/models/'+model_name+'.pkl')
    return model

def load_preprocessor():
    import joblib
    # carga de preprocesador
    preprocessor = joblib.load('/app/models/preprocessor.pkl')
    pca = joblib.load('/app/models/pca.pkl')
    return preprocessor, pca

def predict(input_file, output_file):
    # carga de parametros
    model_name, feature_train = load_params()

    # carga de los datos de entrada
    import pandas as pd
    print('Carga de datos')
    data = pd.read_parquet(input_file)

    # carga de preprocesador y modelo
    print('Carga modelo y preprocesador')
    preprocessor, pca = load_preprocessor()
    model = load_model(model_name)

    # aplicar preprocesador
    X = preprocessor.transform(data)
    if X.shape[1]>feature_train:
        X = pca.transform(X)

    # realizamos predicciones
    print('Realizando predicciones')
    predictions = model.predict(X)

    pd.DataFrame(predictions).to_parquet(output_file)
    print('Predicciones guardadas')
