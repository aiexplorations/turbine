from math import e
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def bivariate_plot(df, first, second, plot_type):
    if first not in df.columns or second not in df.columns:
        raise(Exception)
    
    plot = plt.figure(figsize = (12,12))
    if plot_type == "joint density plot":
        sns.kdeplot(data = df, x = first, y = second)
        plt.title (f"Kernel density plot of {second} vs {first}")
    elif plot_type == "scatter plot":
        sns.scatterplot(data = df, x = first, y = second)
        plt.title (f"Scatter plot of {second} vs {first}")

    return plot