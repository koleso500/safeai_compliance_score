import pandas as pd
from sklearn.base import BaseEstimator
import torch
from typing import Union
from safeai_files.utils import check_nan, convert_to_dataframe, find_yhat
from xgboost import XGBClassifier, XGBRegressor

from safeai_files.core import rga

def compute_rga_parity(xtrain: pd.DataFrame, 
                       xtest: pd.DataFrame, 
                       ytest: list, 
                       yhat: list, 
                       model: Union[XGBClassifier, XGBRegressor, BaseEstimator,
                       torch.nn.Module],
                       protectedvariable: str):
    """
    Compute RGA-based imparity MEASURE. 

    Parameters
    ----------
    xtrain : pd.DataFrame
            A dataframe including train data.
    xtest : pd.DataFrame
            A dataframe including test data.
    ytest : list
            A list of actual values.
    yhat : list
            A list of predicted values.
    model : Union[XGBClassifier, XGBRegressor, BaseEstimator, torch.nn.Module]
            A trained model, which could be a classifier or regressor. 
    protectedvariable: str 
            Name of the protected (sensitive) variable.

    Returns
    -------
    str
            RGA-based imparity score.
    """
    # check if the protected variable is available in data
    if protectedvariable not in xtrain.columns:
        raise ValueError(f"{protectedvariable} is not in the variables")
    xtrain, xtest, ytest, yhat = convert_to_dataframe(xtrain, xtest, ytest, yhat)
    # check for missing values
    check_nan(xtrain, xtest, ytest, yhat)
    # find protected groups
    protected_groups = xtrain[protectedvariable].value_counts().index
    # measure RGA for each group
    rgas = []
    for i in protected_groups:
        xtest_pr = xtest[xtest[protectedvariable]== i]
        ytest_pr = ytest.loc[xtest_pr.index]
        yhat_pr = find_yhat(model, xtest_pr)         
        rga_value = rga(ytest_pr, yhat_pr)
        rgas.append(rga_value)            
    return f"The RGA-based imparity between the protected groups is {max(rgas)-min(rgas)}."
 

