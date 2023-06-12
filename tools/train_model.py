# Script to train machine learning model.
""" 
Train the model.
"""
import joblib
from sklearn.model_selection import train_test_split
from model import train_model, compute_model_metrics, inference
from utils import load_data, process_data, test_slice_data, cat_features

# Add code to load in the data.
DATA_PATH = 'data/census.csv'

if __name__ == "__main__":
    # load data.
    data = load_data(DATA_PATH)
    # split data.
    train, test = train_test_split(data, test_size=0.20)

    # process data.
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, 
        encoder=encoder, lb=lb)

    # Train model.
    print("Training model")
    model = train_model(X_train, y_train)

    print("Score model")
    y_pred = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
    print(f"Precision:{precision:.2f}, Recall:{recall:.2f}, Fbeta:{fbeta:.2f}")

    # test on slices of the data
    print("Computing model metrics on slices of the data")
    test_slice_data(test, cat_features, model, encoder, lb, compute_model_metrics)

    # Save model
    print("Saving model")
    joblib.dump(model, 'model/model.pkl')
    joblib.dump(encoder, 'model/encoder.pkl')
    joblib.dump(lb, 'model/lb.pkl')
    print("Done")