from results import visualize_classification_results, visualize_regression_results
from scipy.sparse import data
from model import display_model_results, fit_model, prepare_training_data
from typing import OrderedDict
from streamlit.errors import Error
from eda import bivariate_plot
from visualize import visualize_data
from sensor_dataframe import generate_stats, return_df_from_db, transform_df
import numpy as np
import pandas as pd
import pandas.api as pandasapi
import streamlit as st
import matplotlib.pyplot as plt



# This is a script to set up a streamlit frontend UI for a data science project
# Using streamlit components such as containers for building up the UI

header_container = st.container()

dataset_base_container = st.container()

dataset_stats_container = st.container()

visualizations_container = st.container()

eda_container = st.container()

transformations_container = st.container()

feature_selection_container = st.container()

model_training_container = st.container()

model_results_container = st.container()

model_saving_container = st.container()

# Retrieve data from database and compute statistics

data_df = return_df_from_db()
metric_columns = ['x1', 'x2', 'y_num', 'y_cat']
sensor_ids = data_df['device_id'].unique().tolist()
sensor_wise_stats = generate_stats(data_df, metric_columns)


# Display on streamlit

with header_container:
    st.markdown("""
    # Turbine: Sensor Data Analytics Application
    A simple application for end-to-end machine learning.
    
    """)

with dataset_base_container:
    st.header('''
    Sensor dataset 
    **Note:**
    * This is a sample of the data present in the database.
    * Only the first few rows are shown.
    ''')
    st.write(data_df.head())

with dataset_stats_container:
    st.header('''
    Sensor-wise dataset summary
    These are sample statistics generated for each sensor
    Key statistics generated are mean, median, standard deviation, kurtosis and skewness.
    ''')
    st.write(sensor_wise_stats)

with visualizations_container:
    st.header('''
    Visualizations of the sensor data - univariate analysis
    ''')
    sensor, metric, kind = st.columns([1,1,1])
    sensor_ids.sort()
    sensor = sensor.selectbox(f"Specify the sensor to be visualized",
                    sensor_ids)
    metric = metric.selectbox(f"Specify the metric to be visualized",
                    metric_columns)
    kind =  kind.selectbox(f"Specify the kind of visualization",
                    ("run chart", "histogram"))
    plots = visualize_data(data_df, metric_columns, kind)

    st.subheader(f"Visualization for sensor id:")
    st.pyplot(plots[str(sensor).zfill(3)+"_"+metric+"_"+kind])

with eda_container:
    st.header('''
    Exploratory Data Analysis
    This section can display joint density plots, and other kinds of charts
    Correlation analysis may also be performed here.
    ''')

    sel_column, disp_column = st.columns([1,2])

    eda_type = sel_column.selectbox(f"Specify kind of bivariate plot",
                            ("joint density plot", "scatter plot"))
    first = sel_column.selectbox(f"Specify first variable",
                            metric_columns)
    second = sel_column.selectbox(f"Specify second variable",
                            metric_columns)

    if first == second:
        st.write("Warning: Different variables should be selected to see a bivariate analysis")
    else:
        #disp_column.subheader(f"A {eda_type} visualization for {second} vs {first}")
        eda_plot = bivariate_plot(data_df, first, second, eda_type)
        disp_column.pyplot(eda_plot)


with transformations_container:
    st.header("Feature engineering options")
    transforms = {}
    for col in metric_columns:
        transformation = st.selectbox(f"Select the transform for {col}", 
                                      ("none", "log", "sqrt", "exp"))
        transforms.update({col: transformation})
    
    na_handling_methods = ["Fill zeros where NA", "Drop NA values"]

    na_handling = st.selectbox("Choose NA values handling method",
                na_handling_methods)

    transformed_df, new_metric_columns = transform_df(data_df, transforms, na_handling)
    st.subheader("Transformed dataframe (sample)")
    st.write(transformed_df.sample(10))
        
with feature_selection_container:
    st.header("Feature Selection for Model Training")
    target_name = st.selectbox(
        "Select the target for the supervised learning model",
        new_metric_columns,
    )
    selected_feature_names = st.multiselect(
        "Select the input features for the model",
        new_metric_columns
    )

    if target_name in selected_feature_names:
        raise(Error("Target should not be present in list of input features"))
    
with model_training_container:
    st.header("Model training options")
    model_types = ["Regression", "Classification", "Clustering"]
    
    # lists of supported algorithms
    # future support for deep learning and other algorithms - may be pre-trained models, or custom models
    # In either case, the same approach to specifying algorithm, parameters has to be maintained

    regression_algorithms = [
        "OLS Regression",
        "Ridge Regression",
        "Lasso Regression",
        "Elastic Net Regression",
        "Decision Tree Regression",
        "Support Vector Regression"
    ]
    classification_algorithms = [
        "Decision Tree Classifier",
        "Support Vector Machine",
        "Logistic Regression",
        "Gaussian Naive Bayes Classifier"
    ]


    # lists of parameters appropriate for each of the algorithm types
    # Ordered dictionaries used to organize hyperparam lists
    # Where we use 2-tuples, we specify upper and lower limits
    # Where we use lists, we specify categorical options

    reg_model_parameter_lists = {

        "OLS Regression":               OrderedDict({
            # no hyper-parameters for OLS regression
        }),
        
        "Ridge Regression":         OrderedDict({
            "alpha":(0.0,10.0)
            }),
        
        "Lasso Regression":         OrderedDict({
            "alpha":(0.0,10.0)
        }),
        
        "Elastic Net Regression":       OrderedDict({
            "alpha":(0.0,10.0), 
            "l1_ratio": (0.0, 1.0)
            }),
        
        "Decision Tree Regression":     OrderedDict(
            {"criterion": ["mse", "friedman_mse", "mae", "poisson"],
            "max_depth": (2, 10),
            "min_samples_split": (20,100),
            "min_samples_leaf": (10, 50)
            }),
        
        "Support Vector Regression":    OrderedDict(
            {"kernel": ["linear", "poly", "rbf", "sigmoid", "precomputed"],
            "degree": (3),
            "C": (0.001, 10.0), 
            }),

    }

    clf_model_parameter_lists = {
        "Decision Tree Classifier":     OrderedDict({
            "criterion": ["gini", "entropy"],
                        "max_depth": (2, 10),
            "min_samples_split": (20,100),
            "min_samples_leaf": (10, 50)
        }),
        
        "Support Vector Machine":       OrderedDict({
            "kernel": ["linear", "poly", "rbf", "sigmoid", "precomputed"],
            "degree": (3),
            "C": (0.001, 10.0), 
        }),
        
        "Logistic Regression":          OrderedDict({
            "penalty": ["l1", "l2", "elastic_net", "none"],
            "C": (0.0, 10.0),
            "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
            "l1_ratio": (0,1),
            "max_iter": (10,1000),
            "multi_class": ["auto"],
            "n_jobs": [-1], #uses all cores on disk
        }),
        
        "Gaussian Naive Bayes Classifier":       OrderedDict({
                # Takes no hyperparameters
        })
    }


    # Retrieving the appropriate kind of model including algorithm input from the front end
    if pandasapi.types.is_numeric_dtype(data_df[target_name]):
        st.write("Target variable is numeric type. Regression models may be built.")
        selected_model_type = "Regression"
    elif not pandasapi.types.is_numeric_dtype(data_df[target_name]):
        st.write("Target variable is not of numeric type. Classification models may be built.")
        selected_model_type = "Classification"
    
    if selected_model_type not in ["Regression", "Classification"]:
        raise(Exception("Only Regression and Classification models supported"))
    if selected_model_type == "Regression":
        algorithm_type = st.selectbox("Choose the kind of algorithm you wish to train",
                                regression_algorithms)
    elif selected_model_type == "Classification":
        algorithm_type =  st.selectbox("Choose the kind of algorithm you wish to train",
                                classification_algorithms)

    if len(reg_model_parameter_lists[algorithm_type]) > 0:
        st.subheader("Provide model and data parameters for this algorithm:")
    

    # Obtain model hyperparameters for the chosen algorithm
    model_hyperparams = {}
    
    for param, param_range in reg_model_parameter_lists[algorithm_type].items():
        if type(param_range) == list:
            model_hyperparams.update({param: st.selectbox(f"Enter the value of {param}",
                                                            param_range)})
        elif type(param_range) == tuple:
            model_hyperparams.update({param: st.slider(f"Enter the value of {param}",
                                                            min_value=param_range[0], max_value=param_range[1])})
    
    
    st.write(f"Model hyperparameters obtained are: {model_hyperparams}")

    model_params = {
        
        "model_type": selected_model_type,
        "target_name": target_name,
        "selected_feature_names": selected_feature_names,
        "model_type": algorithm_type,
        "model_hyperparams": model_hyperparams,
    }

    data_params = {
        "cv_type": "hold_out_cv", #k-fold-cv not implemented
        "is_scaled_data": False,
    }



    data_params.update({"test_frac": st.slider("Provide the fraction of the data to be used for testing",
                                        min_value=0.1, max_value=0.5)})
    
    data_params.update({"scaler_type": st.selectbox("Which scaler would you like to use?",
                                        ["Min Max Scaler", "Standard Scaler"])})
    
    st.subheader("Preparation of training and test datasets")

    #st.write(model_params)
    #st.write(data_params)

    is_data_prepared = False

    if st.checkbox("Prepare datasets"):
        st.write("Preparing training and test data...")
        prepared_data = prepare_training_data(data=transformed_df, model_params=model_params, data_params=data_params)
        st.write("Dataset prepared for model training and testing")
        is_data_prepared = True
    
    print(is_data_prepared)

    is_model_trained = False
    if st.checkbox("Train model"):
        if is_data_prepared == False:
            raise(Exception("Data should be prepared, kindly click on 'prepare data' button above"))
        
        #print("Entering training loop")
        st.write("Training model...")
        model_training_results = fit_model(data=prepared_data, model_params=model_params, data_params=data_params)
        st.write("Model training complete")
        is_model_trained = True
    
    print(is_model_trained)

with model_results_container:
    if is_model_trained == True:
        if st.button("Model performance metrics"):
            y_true = prepared_data['yts']
            xts = prepared_data["xts"]
            y_pred = model_training_results["model"].predict(xts)
            st.write("Results for test set")
            st.write(display_model_results(model_type=selected_model_type, y_true=y_true, y_pred=y_pred))
        if st.button("Model visualization"):
            # Add logic for visualization of model performance
            # Regression - Residual Plot, Prediction Error Plot
            # Classification - ROC-AUC, Precision-Recall Curve, Classification Report
            if selected_model_type == "Regression":
                results = visualize_regression_results(prepared_data, model_training_results["model"])
                for result in results:
                    st.pyplot(result.show())
            elif selected_model_type == "Classification":
                results = visualize_classification_results(prepared_data, model_training_results["model"])
                for result in results:
                    st.pyplot(result.show())

