from operator import mod
from numpy.core.numeric import NaN
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from streamlit.errors import Error
import sklearn.metrics as metrics


def prepare_training_data(data, model_params, data_params):
    """
    This function takes a dataframe and prepares a train/test dataset with it.
    Model and data params provided indicate model type, target column, test set fraction and other details
    """

    model_type = model_params["model_type"]
    target_name = model_params["target_name"]
    selected_feature_names = model_params["selected_feature_names"]
    cv_type = data_params["cv_type"]
    test_frac = data_params["test_frac"]
    is_scaled = data_params["is_scaled_data"]

    # print(is_scaled, cv_type)
    if is_scaled == True:
        raise(Exception("Pass unscaled data only"))
    
    if (is_scaled == False) and (cv_type in ["hold_out_cv", "k_fold_cv"]):
        
        X, y = data[selected_feature_names], data[target_name]
        xtr, xts, ytr, yts = train_test_split(X, y, test_size = test_frac)
    else:
        raise(Error("Incorrect model parameter combinations"))

    dataset = {
        "xtr": xtr,
        "xts": xts,
        "ytr": ytr,
        "yts": yts
    }
    return dataset


def fit_model(data, model_params, data_params):
    # Identify the model type and parameters
    print("Fitting model")
    model_type = model_params['model_type']
    model_hyperparams = model_params['model_hyperparams']
    
    # Extract data from the dataset passed
    xtr, xts, ytr, yts = data['xtr'], data['xts'], data['ytr'], data['yts']

    if data_params["is_scaled_data"] == False:
        if data_params["scaler_type"] == "Min Max Scaler":
            scaler = MinMaxScaler()
        elif data_params["scaler_type"] == "Standard Scaler":
            scaler = StandardScaler()
    
        xtr_sc = scaler.fit_transform(xtr)
        xts_sc = scaler.transform(xts)
    elif data_params["is_scaled_data"] == True:
        xtr_sc, xts_sc = xtr, xts

    if model_type == "OLS Regression":
        model = LinearRegression()
        
    elif model_type == "Ridge Regression":
        model = Ridge(alpha=model_hyperparams["alpha"])
    
    elif model_type == "Lasso Regression":
        model = Lasso(alpha=model_hyperparams["alpha"])

    elif model_type == "Elastic Net Regression":
        model = ElasticNet(
            alpha=model_hyperparams["alpha"], 
            l1_ratio=model_hyperparams['l1_ratio']
            )

    elif model_type == "Decision Tree Regression":
        model = DecisionTreeRegressor(
            criterion=model_hyperparams['criterion'],
            max_depth=model_hyperparams['max_depth'],
            min_samples_split=model_hyperparams['min_samples_split'],
            min_samples_leaf=model_hyperparams['min_samples_leaf']
        )

    elif model_type == "Support Vector Regressor":
        model = SVR(
            kernel=model_hyperparams["kernel"], 
            degree=model_hyperparams["degree"], 
            C=model_hyperparams["C"]
        )

    elif model_type == "Decision Tree Classifier":
        model = DecisionTreeClassifier(
            criterion=model_hyperparams['criterion'],
            max_depth=model_hyperparams['max_depth'],
            min_samples_split=model_hyperparams['min_samples_split'],
            min_samples_leaf=model_hyperparams['min_samples_leaf']
        )
    elif model_type == "Support Vector Machine":
        model = SVC(
            kernel=model_hyperparams["kernel"], 
            degree=model_hyperparams["degree"], 
            C=model_hyperparams["C"]
        )

    elif model_type == "Logistic Regression":
        model = LogisticRegression(
            penalty=model_hyperparams["penalty"],
            C=model_hyperparams["C"],
            solver=model_hyperparams["solver"],
            l1_ratio=model_hyperparams["l1_ratio"],
            max_iter=model_hyperparams["max_iter"],
            multi_class=model_hyperparams["multi_class"],
            n_jobs=model_hyperparams["n_jobs"]
        )
    
    elif model_type == "Gaussian Naive Bayes Classifier":
        model = GaussianNB() #Gaussian Naive Bayes takes no parameters
    
    # Train the defined machine learning model on the data
    model.fit(xtr_sc, ytr)
    print("Model fit complete")

    train_score = model.score(xtr_sc, ytr)
    test_score = model.score(xts_sc, yts)
    result = {
            "model" : model,
            "train_score": train_score,
            "test_score": test_score,
    }

    return result
    

def display_model_results(model_type, y_true, y_pred):
    
    # Regression metrics function from https://stackoverflow.com/a/57239611 
    if model_type == "Regression":
        explained_variance=metrics.explained_variance_score(y_true, y_pred)
        mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred)
        mse=metrics.mean_squared_error(y_true, y_pred)
        if min(y_pred) > 0 and min(y_true) > 0:
            mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
        else:
            mean_squared_log_error=NaN
        median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
        r2=metrics.r2_score(y_true, y_pred) 

        results = {
            'explained_variance': round(explained_variance,4),
            'mean_squared_log_error': round(mean_squared_log_error,4),
            'r2': round(r2,4),
            'MAE': round(mean_absolute_error,4),
            'MSE': round(mse,4),
            'RMSE': round(np.sqrt(mse),4)
        }

    # Constructed classification metrics from sklearn.metrics 
    elif model_type=="Classification":
        precision = metrics.precision_score(y_true, y_pred)
        recall =    metrics.recall_score(y_true, y_pred)
        f1_score = metrics.f1_score(y_true, y_pred)

        results = {
            "precision score": round(precision, 2),
            "recall score": round(recall,2),
            "f1 score": round(f1_score, 2)
        }
        
    return pd.DataFrame.from_records(results, index = [0])


