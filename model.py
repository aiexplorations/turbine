import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR


def define_model(data, model_params):
    # Identify the model type and parameters
    model_type = model_params['model_type']
    model_hyperparams = model_params['model_hyperparams']
    
    # Extract data from the dataset passed
    xtr, xts, ytr, yts = data['xtr'], data['xts'], data['ytr'], data['yts']

    if model_type == "Linear Regression":
        model = LinearRegression()
        model.fit(xtr, ytr)
        
    elif model_type == "Ridge Regression":
        model = Ridge(alpha=model_hyperparams["alpha"])
        model.fit(xtr, ytr)
    
    elif model_type == "Lasso Regression":
        model = Lasso(alpha=model_hyperparams["alpha"])
        model.fit(xtr, ytr)

    elif model_type == "Elastic Net Regression":
        model = DecisionTreeRegressor(
            criterion=model_hyperparams['criterion'],
            max_depth=model_hyperparams['max_depth'],
            min_samples_split=model_hyperparams['min_samples_split'],
            min_samples_leaf=model_hyperparams['min_samples_leaf']
        )
        model.fit(xtr, ytr)

    elif model_type == "Decision Tree Regression":
        model = DecisionTreeRegressor(
            criterion=model_hyperparams['criterion'],
            max_depth=model_hyperparams['max_depth'],
            min_samples_split=model_hyperparams['min_samples_split'],
            min_samples_leaf=model_hyperparams['min_samples_leaf']
        )
        model.fit(xtr, ytr)

    train_score = model.score(xtr, xts)
    test_score = model.score(xts, yts)
    result = {
            "model" : model,
            "train_score": train_score,
            "test_score": test_score,
    }

    return result
    