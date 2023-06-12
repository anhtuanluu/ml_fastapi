# Deploying a Machine Learning Model on Render with FastAPI  

This project develops a classification model on publicly available Census Bureau data. This project will create unit tests to monitor the model performance on various data slices. Then, will deploy your model using the FastAPI package and create API tests. The slice validation and the API tests will be incorporated into a CI/CD framework using GitHub Actions.

### Environment Setup  

* Download and install conda if you don’t have it already.
* Use the supplied requirements file to create a new environment
```
conda create -n [envname] "python=3.8" scikit-learn pandas numpy pytest jupyter jupyterlab fastapi uvicorn -c conda-forge
```
* activate the env
```
conda activate [envname]
```

### Train a model  

* To train the model run:
``` 
python tools/train_model.py
```

* Test the API on local server
```
python main.py
```
* Unit test
```
pytest
```

### GitHub Actions  

The machine learning pipeline is deployed automatically in a CI/CD. After successfully passing all the tests, the code is automaticaly pushed to the Render deployment.

### Test Render deployment  

* Executing a POST request:
```
python live_post.py
```