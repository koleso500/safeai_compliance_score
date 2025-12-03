import io
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.base import BaseEstimator, is_classifier, is_regressor
import torch
from typing import Union
from xgboost import XGBClassifier, XGBRegressor

from src.models import QSVCWrapper


def manipulate_testdata(xtrain: pd.DataFrame, 
                        xtest: pd.DataFrame,
                        variable: str):
    """
    Manipulate the given variable column in test data based on values of that variable in train data.

    Parameters
    ----------
    xtrain : pd.DataFrame
            A dataframe including train data.
    xtest : pd.DataFrame
            A dataframe including test data.
    variable: str 
            Name of variable.

    Returns
    -------
    pd.DataFrame
            The manipulated data.

    """
    xtest_rm = xtest.copy()

    # Detect categorical variable
    if isinstance(xtrain[variable].dtype, pd.CategoricalDtype) or xtrain[variable].dtype == object:
        # Replace category with mode
        mode_value = xtrain[variable].mode(dropna=True)[0]
        xtest_rm[variable] = mode_value
    else:
        # Replace numeric variable with mean
        mean_value = xtrain[variable].mean()
        xtest_rm[variable] = mean_value

    return xtest_rm


def convert_to_dataframe(*args):
    """
    Convert inputs to DataFrames.

    Parameters
    ----------
    *args
            A variable number of input objects that can be converted into Pandas DataFrames (e.g., lists, dictionaries, numpy arrays).

    Returns
    -------
    list of pd.DataFrame
            A list of Pandas DataFrames created from the input objects.    
    """
    return [pd.DataFrame(arg).reset_index(drop=True) for arg in args]


def validate_variables(variables: Union[list, str], xtrain: pd.DataFrame):
    """
    Check if variables are valid and exist in the train dataset.

    Parameters
    ----------
    variables: list or str
            Variables.
    xtrain : pd.DataFrame
            A dataframe including train data.

    Raises
    -------
    ValueError
            If variables is not a list, not a string or if any variable does not exist in xtrain.
    """
    if isinstance(variables, str):
        variables = [variables]
    elif not isinstance(variables, list):
        raise ValueError("Variables input must be a list")
    for var in variables:
        if var not in xtrain.columns:
            raise ValueError(f"{var} is not in the variables")


def check_nan(*dataframes):
    """
    Check if any of the provided DataFrames contain missing values.

    Parameters
    ----------
    *dataframes : pd.DataFrame
        A variable number of DataFrame objects to check for NaN values.

    Raises
    ------
    ValueError
        If any DataFrame contains missing (NaN) values.
    TypeError
        If any input is not a Pandas DataFrame.
    """

    for i, df in enumerate(dataframes, start=1):
        if isinstance(df, pd.DataFrame):  # Ensure df is a DataFrame
            if df.isna().sum().sum() > 0:  # Check if there are any missing values
                raise ValueError(f"DataFrame {i} contains missing values")
        else:
            raise TypeError(f"Item {i} is not a pandas DataFrame")



def find_yhat(model: Union[XGBClassifier, XGBRegressor, BaseEstimator,
              torch.nn.Module],
              xtest: pd.DataFrame):
    """
    Find predicted values for the manipulated data.

    Parameters
    ----------
    xtest : pd.DataFrame
            A dataframe including test data.
    model : Union[XGBClassifier, XGBRegressor, BaseEstimator, torch.nn.Module]
            A trained model, which could be a classifier or regressor.

    Returns
    -------
            The yhat value.
    """
    if isinstance(model, QSVCWrapper):
        return model.predict_proba(xtest)[:, 1]
    if is_classifier(model):
        yhat = [x[1] for x in model.predict_proba(xtest)]
    elif is_regressor(model):
        yhat = model.predict(xtest)
    elif isinstance(model, torch.nn.Module):
        xtest_tensor = torch.tensor(xtest.values, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            yhat = model(xtest_tensor)
        if yhat.shape[1] == 2:  # binary classification
            yhat = yhat[:, 1].numpy()
        else:
            yhat = yhat.numpy()
    elif hasattr(model, "predict_proba"):
        return model.predict_proba(xtest)[:, 1]
    else:
        raise ValueError("The model type is not recognized for prediction")

    return yhat


def plot_model_curves(x, curves, model_name, prefix="Curve", title="", xlabel='Steps', ylabel='Values', ax=None):
    """
    Plot RGA/RGE/RGR curves for a given model using a template.

    Parameters:
    x (np.ndarray or list): X-axis values.
    curves (list of np.ndarray): List of curves to plot (e.g., [rga, rge, rgr])
    model_name (str): Name of the model (e.g., "RF", "XGB") used for labeling
    prefix (str): Prefix for legend labels (e.g., "Curve", "Difference Random")
    title (str): Title of the plot
    xlabel (str): Label for the x-axis
    ylabel (str): Label for the y-axis
    """
    labels_base = ["RGA", "RGE", "RGR"]
    labels = [f"{label} {prefix} {model_name}" for label in labels_base]

    if model_name == "Random":
        line_styles = ['-', '--', ':']  # Make each line style different
    else:
        line_styles = ['-'] * len(curves)

    if ax is None:
        plt.figure(figsize=(6, 4))
        for curve, label, style in zip(curves, labels, line_styles):
            plt.plot(x, curve, linestyle=style, label=label)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.xlim([0, 1])
        plt.grid(True)
    else:
        for curve, label, style in zip(curves, labels, line_styles):
            ax.plot(x, curve, linestyle=style, label=label)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim([0, 1])
        ax.grid(True)
        ax.legend()


def plot_metric_distribution(metric_values, print_label, xlabel, title, bar_label="Model", bins=60, ax=None):
    """
    Plot a histogram for the given metric values with normalized counts.

    Parameters:
        metric_values (np.ndarray): Computed metric values
        print_label (str): Label to print for the mean volume
        xlabel (str): Label for the x-axis of the plot
        title (str): Title for the histogram plot
        bar_label (str): Label in the legend
        bins (int): Number of bins for the histogram
    """
    # Flatten and compute mean
    flat_vals = metric_values.flatten()
    total_sum = np.sum(flat_vals)
    num_elements = flat_vals.size
    normalized_volume = total_sum / num_elements

    # Print mean volume
    print(f"{print_label}: {normalized_volume:.5f}")

    # Histogram
    counts, bin_edges = np.histogram(flat_vals, bins=bins)
    max_count = counts.max()
    counts_norm = counts / max_count
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Plot
    if ax is None:
        plt.figure(figsize=(10, 6))
        plt.bar(bin_centers, counts_norm, width=(bin_edges[1] - bin_edges[0]), alpha=0.7, label=bar_label)
        plt.xlabel(xlabel)
        plt.ylabel('Normalized Counts')
        plt.title(title)
        plt.grid(True)
        plt.legend()
    else:
        ax.bar(bin_centers, counts_norm, width=(bin_edges[1] - bin_edges[0]), alpha=0.7, label=bar_label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Normalized Counts')
        ax.set_title(title)
        ax.grid(True)
        ax.legend()

def plot_metric_distribution_diff(metric_values, print_label, xlabel, title, bar_label="Model", bins=60, ax=None):
    """
    Plot a histogram for the given metric differences values with normalized counts.

    Parameters:
        metric_values (np.ndarray): Computed metric values
        print_label (str): Label to print for the mean volume
        xlabel (str): Label for the x-axis of the plot
        title (str): Title for the histogram plot
        bar_label (str): Label in the legend
        bins (int): Number of bins for the histogram
    """
    # Flatten and compute mean
    flat_vals = metric_values.flatten()
    total_sum = np.sum(flat_vals)
    num_elements = flat_vals.size
    normalized_volume = total_sum / num_elements

    # Print mean volume
    print(f"{print_label}: {normalized_volume:.5f}")

    # Histogram
    counts, bin_edges = np.histogram(flat_vals, bins=bins)
    max_count = counts.max()
    counts_norm = counts / max_count
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Plot
    if ax is None:
        plt.figure(figsize=(10, 6))
        plt.bar(bin_centers, counts_norm, width=(bin_edges[1] - bin_edges[0]), alpha=0.7, label=bar_label, color='green')
        plt.axvline(0, color='red', linestyle='--', label='No Difference')
        plt.xlabel(xlabel)
        plt.ylabel('Normalized Counts')
        plt.title(title)
        plt.grid(True)
        plt.legend()
    else:
        ax.bar(bin_centers, counts_norm, width=(bin_edges[1] - bin_edges[0]), alpha=0.7, label=bar_label)
        ax.axvline(0, color='red', linestyle='--', label='No Difference')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Normalized Counts')
        ax.set_title(title)
        ax.grid(True)
        ax.legend()

def plot_mean_histogram(rga, rge, rgr, *,
                        model_name: str,
                        bar_label: str,
                        mean_type: str,
                        ax = None):
    """
    Compute different means (arithmetic, geometric, quadratic) of (rga, rge, rgr),
    then call plot_metric_distribution_diff with the appropriate labels.

    Parameters
    ----------
    rga, rge, rgr : array‐like
        Three metrics for this model.
    model_name : str
        Name used inside the “print_label” (e.g. "Logistic Regression").
    bar_label : str
        The text to show under each bar in the histogram legend.
    mean_type : str
        Mean formula.
    """
    if mean_type == "arithmetic":
        values = (rga + rge + rgr) / 3
        print_label = f"Mean volume Arithmetic {model_name}"
        xlabel = "Normalized Arithmetic Mean"
        title = f"Histogram of Normalized Arithmetic Mean Values ({model_name})"
    elif mean_type == "geometric":
        values = np.cbrt(rga * rge * rgr)
        print_label = f"Mean volume Geometric {model_name}"
        xlabel = "Normalized Geometric Mean (1/3)"
        title = f"Histogram of Normalized Geometric Mean (1/3) Values ({model_name})"
    elif mean_type == "quadratic":
        values = np.sqrt((rga ** 2 + rge ** 2 + rgr ** 2) / 3)
        print_label = f"Mean volume Quadratic Mean (RMS) {model_name}"
        xlabel = "Normalized Quadratic Mean (RMS)"
        title = f"Histogram of Normalized Quadratic Mean (RMS) Values ({model_name})"
    else:
        raise ValueError("`mean_type` is not added yet")

    plot_metric_distribution(
        metric_values=values,
        print_label=print_label,
        xlabel=xlabel,
        title=title,
        bar_label=bar_label,
        ax=ax
    )

def plot_diff_mean_histogram(rga, rge, rgr, *,
                        model_name: str,
                        bar_label: str,
                        mean_type: str,
                        ax = None):
    """
    Compute different means of differences values with base model (arithmetic, geometric, quadratic) of (rga, rge, rgr),
    then call plot_metric_distribution_diff with the appropriate labels.

    Parameters
    ----------
    rga, rge, rgr : array‐like
        Three metrics for this model.
    model_name : str
        Name used inside the “print_label” (e.g. "Logistic Regression").
    bar_label : str
        The text to show under each bar in the histogram legend.
    mean_type : str
        Mean formula.
    """
    if mean_type == "arithmetic":
        values = (rga + rge + rgr) / 3
        print_label = f"Difference Arithmetic {model_name}"
        xlabel = "Normalized Difference Arithmetic Mean"
        title = f"Histogram of Difference Arithmetic Mean Values  ({model_name})"
    elif mean_type == "geometric":
        values = np.cbrt(rga * rge * rgr)
        print_label = f"Difference Geometric Mean (1/3) {model_name}"
        xlabel = "Normalized Difference Geometric Mean (1/3)"
        title = f"Histogram of Difference Geometric Mean (1/3) Values ({model_name})"
    elif mean_type == "quadratic":
        values = np.sqrt((rga ** 2 + rge ** 2 + rgr ** 2) / 3)
        print_label = f"Difference Mean volume Quadratic Mean (RMS) {model_name}"
        xlabel = "Normalized Difference Quadratic Mean (RMS)"
        title = f"Histogram of Difference Quadratic Mean (RMS) Values ({model_name})"
    else:
        raise ValueError("`mean_type` is not added yet")

    plot_metric_distribution_diff(
        metric_values=values,
        print_label=print_label,
        xlabel=xlabel,
        title=title,
        bar_label=bar_label,
        ax=ax
    )

def save_model_metrics(result, save_dir="results_metrics"):
    os.makedirs(save_dir, exist_ok=True)
    model_name = result['model_name']
    filepath = os.path.join(save_dir, f"{model_name}_results.json")
    f: io.TextIOWrapper
    with open(filepath, 'w') as f:
        json.dump(result, f, indent=4)