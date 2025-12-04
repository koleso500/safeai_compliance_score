import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
import torch
from typing import Union, Literal
from safeai_files.utils import check_nan, convert_to_dataframe, find_yhat, validate_variables
from xgboost import XGBClassifier, XGBRegressor

from safeai_files.core import rga


def perturb(data: pd.DataFrame, 
            variable: str, 
            perturbation_percentage= 0.05):
    """
    Function to perturb a single variable based on the replacement of the two percentiles 
    selected using the perturbation_percentage of the object.

    Parameters
    ----------
    data : pd.DataFrame
            A dataframe including data.
    variable: str 
            Name of variable.
    perturbation_percentage: float
            A percentage value for perturbation. 

    Returns
    -------
    pd.DataFrame
            The perturbed data.
    """ 
    if perturbation_percentage > 0.5 or perturbation_percentage < 0:
        raise ValueError('The perturbation percentage should be between 0 and 0.5.')
        
    data = data.reset_index(drop=True)
    perturbed_variable = data.loc[:,variable]
    vals = [[i, values] for i, values in enumerate(perturbed_variable)]
    indices = [x[0] for x in sorted(vals, key= lambda item: item[1])]
    sorted_variable = [x[1] for x in sorted(vals, key= lambda item: item[1])]

    percentile_5_index = int(np.ceil(perturbation_percentage * len(sorted_variable)))
    percentile_95_index = int(np.ceil((1-perturbation_percentage) * len(sorted_variable)))
    values_before_5th_percentile = sorted_variable[:percentile_5_index]
    values_after_95th_percentile = sorted_variable[percentile_95_index:]
    n = min([len(values_before_5th_percentile), len(values_after_95th_percentile)])
    lowertail_indices = indices[0:n]
    uppertail_indices = (indices[-n:])
    uppertail_indices = uppertail_indices[::-1]
    new_variable = perturbed_variable.copy()

    for j in range(n):
        new_variable[lowertail_indices[j]] = perturbed_variable[uppertail_indices[j]]
        new_variable[uppertail_indices[j]] = perturbed_variable[lowertail_indices[j]]
    data.loc[:,variable] = new_variable
    return data


def compute_rgr_values(xtest: pd.DataFrame, 
                                yhat: list, 
                                model: Union[XGBClassifier, XGBRegressor, BaseEstimator,
                                torch.nn.Module],
                                variables: list, 
                                perturbation_percentage= 0.05):
    """
    Compute RANK GRADUATION Robustness (RGR) MEASURE for single variable contribution.

    Parameters
    ----------
    xtest : pd.DataFrame
            A dataframe including test data.
    yhat : list
            A list of predicted values.
    model : Union[XGBClassifier, XGBRegressor, BaseEstimator, torch.nn.Module]
            A trained model, which could be a classifier or regressor. 
    variables: list 
            A list of variables.
    perturbation_percentage: float
            A percentage value for perturbation .

    Returns
    -------
    pd.DataFrame
            The RGR value.
    """
    # Convert inputs to DataFrames and concatenate them
    xtest, yhat = convert_to_dataframe(xtest, yhat)
    # check for missing values
    check_nan(xtest, yhat)
    # variables should be a list
    validate_variables(variables, xtest)
    # find RGRs
    rgr_list = []
    for i in variables:
        xtest_pert = perturb(xtest, i, perturbation_percentage)
        yhat_pert = find_yhat(model, xtest_pert)
        rgr_list.append(rga(yhat, yhat_pert))
    rgr_df = pd.DataFrame(rgr_list, index= list(variables), columns=['RGR']).sort_values(by='RGR', ascending=False)
    return rgr_df


def rgr_single(xtest: pd.DataFrame,
                yhat: list,
                model: Union[XGBClassifier, XGBRegressor, BaseEstimator, torch.nn.Module],
                variable: str,
                perturbation_percentage=0.05):
    """
    Compute RANK GRADUATION Robustness (RGR) MEASURE for a single variable.

    Parameters
    ----------
    xtest : pd.DataFrame
            A dataframe including test data.
    yhat : list
            A list of predicted values.
    model : Union[XGBClassifier, XGBRegressor, BaseEstimator, torch.nn.Module]
            A trained model, which could be a classifier or regressor.
    variable : str
            The variable (column) in xtest to be perturbed.
    perturbation_percentage: float
            A percentage value for perturbation.

    Returns
    -------
    float
            The RGR value for the specified variable.
    """
    # Convert inputs to DataFrames and concatenate them
    xtest, yhat = convert_to_dataframe(xtest, yhat)

    # Check for missing values
    check_nan(xtest, yhat)

    # Variables should be a list
    validate_variables(variable, xtest)

    # Perturb only the selected variable
    xtest_pert = xtest.copy()
    xtest_pert[variable] = perturb(xtest_pert, variable, perturbation_percentage)[variable]

    # Get perturbed predictions
    yhat_pert = find_yhat(model, xtest_pert)

    # Compute and return RGR value for the selected variable
    return rga(yhat, yhat_pert)



def rgr_all(xtest: pd.DataFrame,
            yhat: list,
            model: Union[XGBClassifier,
            XGBRegressor, BaseEstimator, torch.nn.Module],
            perturbation_percentage = 0.05,
            perturbation_strategy: Literal['percentile_swap', 'gaussian_noise'] = 'percentile_swap',
            seed: int = None,
            scaler = None,
            xtest_original: pd.DataFrame = None):
    """
    Compute RANK GRADUATION Robustness (RGR) MEASURE for all variables simultaneously.

    Parameters
    ----------
    xtest : pd.DataFrame
        A dataframe including test data.
    yhat : list
        A list of predicted values.
    model : Union[XGBClassifier, XGBRegressor,
                  BaseEstimator, torch.nn.Module]
        A trained model, which could be a classifier or regressor.
    perturbation_percentage : float
        A percentage value for perturbation intensity.
        - For 'percentile_swap': percentage of tail values to swap
        - For 'gaussian_noise': noise standard deviation as fraction of entire data std
    perturbation_strategy : str
        - 'percentile_swap': Swap tail percentiles
        - 'gaussian_noise': Add Gaussian noise to data
    seed : int
        Random seed for reproducibility (only with 'gaussian_noise')
    scaler :
        Use if features are scaled. Noise is added in original scale then re-scaled.
    xtest_original : pd.DataFrame
        Original unscaled test data. Required when scaler.

    Returns
    -------
    float
        The overall RGR value.

    """
    # Convert inputs to DataFrames and concatenate them
    xtest, yhat = convert_to_dataframe(xtest, yhat)

    # Check for missing values
    check_nan(xtest, yhat)

    # Get all variables in xtest
    variables = xtest.columns.tolist()

    # Perturb based on strategy
    if perturbation_strategy == 'percentile_swap':
        xtest_pert = xtest.copy()
        for var in variables:
            xtest_pert[var] = perturb(xtest_pert, var, perturbation_percentage)[var]

    elif perturbation_strategy == 'gaussian_noise':
        if seed is not None:
            np.random.seed(seed)

        if scaler is not None and xtest_original is not None:
            # Add noise in original scale, then re-scale
            base_std = np.std(xtest_original.values)
            noise = np.random.normal(0.0, perturbation_percentage * base_std,
                                     size=xtest_original.shape).astype(np.float32)

            # Add noise to original data
            xtest_perturbed_original = xtest_original.values + noise

            xtest_perturbed_df = pd.DataFrame(xtest_perturbed_original,
                                              columns=xtest_original.columns,
                                              index=xtest_original.index)

            # Re-scale the perturbed data
            xtest_pert_array = scaler.transform(xtest_perturbed_df)

        else:
            # Calculate base std
            base_std = np.std(xtest.values)

            # Generate Gaussian noise
            noise = np.random.normal(0.0, perturbation_percentage * base_std,
                                    size=xtest.shape).astype(np.float32)

            # Add noise to features
            xtest_pert_array = xtest.values + noise

        # Convert back to DataFrame
        xtest_pert = pd.DataFrame(xtest_pert_array,
                                  columns=xtest.columns,
                                  index=xtest.index)

    else:
        raise ValueError(f'Unknown perturbation strategy: {perturbation_strategy}.'
                         f"Choose from: 'percentile_swap', 'gaussian_noise'")

    # Get perturbed predictions
    yhat_pert = find_yhat(model, xtest_pert)

    # Compute and return RGR value
    return rga(yhat, yhat_pert)


