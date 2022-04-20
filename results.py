from yellowbrick.regressor import ResidualsPlot, PredictionError, CooksDistance
from yellowbrick.classifier import ROCAUC, PrecisionRecallCurve, ClassificationReport, DiscriminationThreshold


def visualize_regression_results(data, model):
    '''
    Using Scikit Yellowbrick to visualize regression models
    '''


    # constructing residuals plot
    residuals_plot = ResidualsPlot(model)
    residuals_plot.fit(data["xtr"], data["ytr"])
    residuals_plot.score(data["xts"], data["yts"])

    # constructing prediction error plot
    pred_error = PredictionError(model)    
    pred_error.fit(data["xtr"], data["ytr"])
    pred_error.score(data["xts"], data["yts"])

    # constructing cook's distance plot
    cooks_dist = CooksDistance(model)    
    cooks_dist.fit(data["xtr"], data["ytr"])
    cooks_dist.score(data["xts"], data["yts"])

    visualizations = {
        "residuals_plot": residuals_plot,
        "pred_error": pred_error,
        "cooks_dist": cooks_dist
    }

    return visualizations

def visualize_classification_results(data, model):
    '''
    Using Scikit Yellowbrick to visualize classification results
    '''

    # to be implemented
    pass
