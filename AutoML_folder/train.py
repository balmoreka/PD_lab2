def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# cargar .env
import os
from dotenv import load_dotenv
# load_dotenv()

train_file = '/'+os.getenv("DATASET")
target_column = os.getenv("TARGET")
model_name = os.getenv("MODEL")
n_trials = int(os.getenv("TRIALS"))
depl_type = os.getenv("DEPLOYMENT_TYPE")
input_folder = os.getenv("INPUT_FOLDER")
output_folder = os.getenv("OUTPUT_FOLDER")
port = os.getenv('PORT')
feature_train = int(os.getenv("FEATURE_TRAIN"))


def preprocess(input_file, target_column, feature_train, output_file, output_prep, output_pca):
    import pandas as pd
    import numpy as np
    import joblib
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.decomposition import KernelPCA

    df = pd.read_parquet(input_file, engine='pyarrow')

    # separar caracteristicas
    features_name = [x for x in list(df.columns) if x!=target_column]
    features = df[features_name]
    target = df[[target_column]]

    # definición de tipo de features
    numeric_features = []
    categoric_features = []
    for f in features_name:
        if (features[f].nunique()<=20):
            categoric_features.append(f)
        else:
            numeric_features.append(f)

            # Ajuste de features
    feat_num = features[numeric_features].select_dtypes('O')
    for f in feat_num.columns:
        feat_num.loc[(~feat_num[f].str.isnumeric()), f] = np.nan
    feat_num = feat_num.astype('float32')
    feat_num = pd.concat([features[numeric_features].select_dtypes('number'),feat_num], axis=1)
    feat_cat = features[categoric_features]
    features = pd.concat([feat_num, feat_cat], axis=1)

    # preprocesamiento
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categoric_features)
        ]
    )

    df = preprocessor.fit_transform(features)
    pca = KernelPCA(n_components=feature_train, kernel='rbf')

    if df.shape[1]>feature_train:
        df = pca.fit_transform(df)
    print(f'Model will be trained with {df.shape[1]} features')

    # guardar preprocesador
    joblib.dump(preprocessor, output_prep)
    joblib.dump(pca, output_pca)

    df = pd.concat([pd.DataFrame(df), target], axis=1, ignore_index=True)
    df.to_parquet(output_file, index=False, engine='pyarrow')
    print('Preprocessing stage is done')
    return df

def train(n_trials, train_func, metric_file):
    import pandas as pd
    import numpy as np
    import joblib
    from sklearn.model_selection import train_test_split
    train_set = '/app/data/data_prep.parquet'
    dataset = pd.read_parquet(train_set, engine='pyarrow')
    X = dataset.iloc[:, :-1].to_numpy()
    y = dataset.iloc[:,-1].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = train_func(X_train, y_train, n_trials)
    evaluate(model, X_test, y_test, metric_file)
    print('Training stage is done')
    return model

def evaluate(model, X_test, y_test, metric_file):
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    import json
    import numpy as np
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    cf = confusion_matrix(y_test, y_pred)
    metrics = {'accuracy':accuracy,
               'f1':f1,
               'confusion_matrix':np.array2string(cf,separator=',',suppress_small=True)}
    with open(metric_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(metrics)
    print('Metrics were saved in metrics.json')

def trainRandomForest(X_train, Y_train, n_trials):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV
    import numpy as np
    import joblib
    param_grid = {'n_estimators':np.arange(50,100,10),
                  'min_samples_leaf':np.arange(0.1,0.8,0.1)}
    model = RandomizedSearchCV(RandomForestClassifier(), param_grid, cv=5, n_jobs=-1, n_iter=n_trials)
    model.fit(X_train, Y_train)
    final_model = model.best_estimator_
    joblib.dump(final_model, '/app/models/RandomForest.pkl')
    print('Best model have been saved in /app/models/RandomForest.pkl')
    return final_model

def trainSVM(X_train, Y_train, n_trials):
    from scipy.stats import uniform
    from sklearn.svm import SVC
    from sklearn.model_selection import RandomizedSearchCV
    import joblib
    param_grid = {'C':uniform(0,2),
                  'kernel':['rbf','sigmoid']}
    model = RandomizedSearchCV(SVC(), param_grid, cv=5, n_jobs=-1, n_iter=n_trials)
    model.fit(X_train, Y_train)
    final_model = model.best_estimator_
    joblib.dump(final_model, '/app/models/SVM.pkl')
    print('Best model have been saved in /app/models/SVM.pkl')
    return final_model

def trainKNN(X_train, Y_train, n_trials):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import RandomizedSearchCV
    import numpy as np
    import joblib
    param_grid = {'n_neighbors':np.arange(1,10,1),
                  'weights':['uniform','distance'],
                  'algorithm':['auto','ball_tree','kd_tree','brute']}
    model = RandomizedSearchCV(KNeighborsClassifier, param_grid, cv=5, n_jobs=-1, n_iter=n_trials)
    model.fit(X_train, Y_train)
    final_model = model.best_estimator_
    joblib.dump(final_model, '/app/models/KNN.pkl')
    print('Best model have been saved in /app/models/KNN.pkl')
    return final_model

def trainNB(X_train, Y_train, n_trials):
    from sklearn.naive_bayes import GaussianNB
    from sklearn.model_selection import RandomizedSearchCV
    import numpy as np
    import joblib
    param_grid = {'var_smoothing':np.arange(1e-15, 1e-3, 1e-5),}
    model = RandomizedSearchCV(GaussianNB(), param_grid, cv=5, n_jobs=-1, n_iter=n_trials)
    model.fit(X_train, Y_train)
    final_model = model.best_estimator_
    joblib.dump(final_model, '/app/models/NaiveBayes.pkl')
    print('Best model have been saved in /app/models/NaiveBayes.pkl')
    return final_model

def trainGB(X_train, Y_train, n_trials):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import RandomizedSearchCV
    import numpy as np
    import joblib
    param_grid = {'max_depth':np.arange(1,10,1),
                  'n_estimators':np.arange(10,100,1),
                  'min_samples_leaf':np.arange(0.1,0.8,0.1)}
    model = RandomizedSearchCV(GradientBoostingClassifier(), param_grid, cv=5, n_jobs=-1, n_iter=n_trials)
    model.fit(X_train, Y_train)
    final_model = model.best_estimator_
    joblib.dump(final_model, '/app/models/GradientBoosting.pkl')
    print('Best model have been saved in /app/models/GradientBoosting.pkl')
    return final_model


## PREPROCESSING STAGE

output_file = '/app/data/data_prep.parquet'
output_prep = '/app/models/preprocessor.pkl'
output_pca = '/app/models/pca.pkl'
preprocess(train_file, target_column, feature_train, output_file, output_prep, output_pca)

## TRAINING STAGE

model_selection = {'RandomForest':trainRandomForest,
                   'GradientBoosting':trainSVM,
                   'SVM':trainSVM,
                   'KNN':trainKNN,
                   'NaiveBayes':trainNB}

metric_file = '/app/metrics/metrics.json'
train_func = model_selection[model_name]
model = train(n_trials, train_func, metric_file)





