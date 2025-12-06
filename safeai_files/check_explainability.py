import pandas as pd
from sklearn.base import BaseEstimator
import torch
from typing import Union
from safeai_files.utils import check_nan, convert_to_dataframe, find_yhat, manipulate_testdata, validate_variables
from xgboost import XGBClassifier, XGBRegressor

from safeai_files.core import rga
from src.cramer import wrga_cramer

def compute_rge_values(xtrain: pd.DataFrame, 
                xtest: pd.DataFrame,
                yhat: list,
                model: Union[XGBClassifier, XGBRegressor, BaseEstimator,
                torch.nn.Module],
                variables: list, 
                group: bool = False,
                metric: str = 'original'):
    """
    Helper function to compute the RGE values for given variables or groups of variables.

    Parameters
    ----------
    xtrain : pd.DataFrame
            A dataframe including train data.
    xtest : pd.DataFrame
            A dataframe including test data.
    yhat : list
            A list of predicted values.
    model : Union[XGBClassifier, XGBRegressor, BaseEstimator, torch.nn.Module]
            A trained model, which could be a classifier or regressor. 
    variables : list
            A list of variables.
    group : bool
            If True, calculate RGE for the group of variables as a whole; otherwise, calculate for each variable.
    metric: str
            'original': uses RGE
            'cramer': uses WRGE

    Returns
    -------
    pd.DataFrame
            The RGE values for each variable or for the group.
    """
    # Convert inputs to DataFrames and concatenate them
    xtrain, xtest, yhat = convert_to_dataframe(xtrain, xtest, yhat)
    # check for missing values
    check_nan(xtrain, xtest, yhat)
    # variables should be a list
    validate_variables(variables, xtrain)

    if metric not in ['original', 'cramer']:
        raise ValueError("Metric must be 'original' or 'cramer'")

    # find RGEs
    if group:
        # Apply manipulate_testdata iteratively for each variable in the group
        for variable in variables:
            xtest = manipulate_testdata(xtrain, xtest, variable)
        
        # Calculate yhat after manipulating all variables in the group
        yhat_rm = find_yhat(model, xtest)

        # Calculate a single RGE for the entire group except these variables
        if metric == "original":
            rge = rga(yhat, yhat_rm)
        else: # 'cramer'
            rge = wrga_cramer(yhat, yhat_rm)

        return pd.DataFrame([rge], index=[str(variables)], columns=['RGE'])

    else:
        # Calculate RGE for each variable individually
        rge_list = []
        for variable in variables:
            xtest_rm = manipulate_testdata(xtrain, xtest, variable)
            yhat_rm = find_yhat(model, xtest_rm)

            if metric == "original":
                rge_val = 1 - rga(yhat, yhat_rm)
            else:  # "cramer"
                rge_val = 1 - wrga_cramer(yhat, yhat_rm)

            rge_list.append(rge_val)
        
        return pd.DataFrame(rge_list, index=variables, columns=['RGE']).sort_values(by='RGE', ascending=False)

