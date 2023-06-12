"""
utils
"""
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
import numpy as np
import pandas as pd

cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

cols = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education_num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours-per-week",
        "native-country"]

def load_data(path):
    """
    Load data
    Inputs
    ------
    path :  str
            data path
    Returns
    -------
    df : cleaned pd.dataframe
         dataframe
    """
    df =  pd.read_csv(path, skipinitialspace=True)
    clean_df = (df.replace("?", None).dropna())
    return clean_df

def process_data(
    X, categorical_features=[], label=None, training=True, encoder=None, lb=None
):
    """ Process the data that will be used in the pipeline.
    Inputs
    ------
    X : pd.DataFrame
        Dataframe
    categorical_features: list[str]
        List categorical features
    label : str
        Label column
    training : bool
        Indicator if process is in training mode or inference mode
    Returns
    -------
    X : np.array
        Processed data
    y : np.array
        Processed labels
    encoder : OneHotEncoder
        Trained OneHotEncoder
    lb : LabelBinarizer
        Trained LabelBinarizer
    """

    if label:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])
    X_categorical = X[categorical_features].values
    
    X_continuous = X.drop(*[categorical_features], axis=1)
    
    if training:
        encoder = OneHotEncoder(handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical).toarray()
        y = lb.fit_transform(y.values).reshape(-1)
    else:
        X_categorical = encoder.transform(X_categorical).toarray()
        try:
            y = lb.transform(y.values).reshape(-1)
        except:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb

def test_slice_data(test, cat_features, model, encoder, lb, compute_model_metrics):
    """
    script to test on slices of the data
    Inputs
    ------
    test : test pd.DataFrame
        Dataframe
    cat_features: list[str]
        List categorical features
    model : Model
        sklearn model
    encoder : OneHotEncoder
        Trained OneHotEncoder
    lb : LabelBinarizer
        Trained LabelBinarizer
    Returns
    -------
    slice_output.txt
    """    
    with open('slice_output.txt', 'w') as file:
        for feature in cat_features:
            values = test[feature].unique()
            for value in values:
                data_slice = test[test[feature] == value]
                X_test, y_test, _, _ = process_data(data_slice, cat_features,
                    training=False, label="salary", encoder=encoder, lb=lb)
                y_pred = model.predict(X_test)
                precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
                row = f"{feature}:{value}, Precision:{precision:.2f}, Recall:{recall:.2f}, Fbeta:{fbeta:.2f}\n"
                file.write(row)
