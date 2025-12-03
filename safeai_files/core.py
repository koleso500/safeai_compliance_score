import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from safeai_files.utils import check_nan,convert_to_dataframe

def rga(y: list, yhat: list):
        """
        RANK GRADUATION ACCURACY (RGA) MEASURE 
        Function for the RGA measure computation.

        Parameters
        ----------
        y : list
                A list of actual values.
        yhat : list
                A list of predicted values.

        Returns
        -------
        float
                The RGA value.
        """
            
        # Convert inputs to DataFrames and concatenate them
        y, yhat = convert_to_dataframe(y, yhat)
        # check the length
        if y.shape != yhat.shape:
                raise ValueError("y and yhat should have the same shape.")
        df = pd.concat([y, yhat], axis=1)
        df.columns = ["y", "yhat"]
        # check for missing values
        check_nan(y, yhat)
              
        # Rank yhat values
        df['ryhat'] = df['yhat'].rank(method="min")

        # Group by ryhat and calculate mean of y (support)
        support = df.groupby('ryhat')['y'].mean().reset_index(name='support')

        # Merge support back to the original dataframe
        df = pd.merge(df, support, on="ryhat", how="left")

        # Create the rord column by directly assigning 'support' where ryhat matches
        df['rord'] = df['support']
        
        # Sort df by yhat to get correct ordering for ystar
        df = df.sort_values(by="yhat").reset_index(drop=True)

        # Get ystar in the same order as rord in the sorted dataframe
        ystar = df['rord'].values

        # Create the index array i
        i = np.arange(len(df))

        # Calculate conc, dec (descending order of y) and inc (ascending order of y)
        conc = np.sum(i * ystar)
        sorted_y = np.sort(df['y'])  # y sorted in ascending order
        dec = np.sum(i * sorted_y[::-1])  # y sorted in descending order
        inc = np.sum(i * sorted_y)

        # Compute the RGA
        denom = inc - dec
        if denom == 0:
            return 1.0
        result = (conc - dec) / denom

        return result

def partial_rga_with_curves(y: list, yhat: list, lower=0.0, upper=1.0, plot=True):
    """
    RANK GRADUATION ACCURACY (RGA) MEASURE
    Function for computing full and partial RGA value and plotting the Lorenz, Dual Lorenz, and Concordance curves.

    Parameters
    ----------
    y : list
            A list of actual values.
    yhat : list
              A list of predicted values.
    lower : float, optional
        Lower percentile (between 0 and 1) for partial RGA. Default is 0.0 (full range).
    upper : float, optional
        Upper percentile (between 0 and 1) for partial RGA. Default is 1.0 (full range).
    plot : bool, optional
        Whether to plot Lorenz, Dual Lorenz, and Concordance curves. Default is True.
    Returns
    -------
    partial_rga_contribution : float
        The Partial RGA value.
    """
    # Convert inputs to DataFrames and concatenate them
    y, yhat = convert_to_dataframe(y, yhat)
    # check the length
    if y.shape != yhat.shape:
        raise ValueError("y and yhat should have the same shape.")
    df = pd.concat([y, yhat], axis=1)
    df.columns = ["y", "yhat"]
    # check for missing values
    check_nan(y, yhat)

    # Rank yhat values
    df['ryhat'] = df['yhat'].rank(method="min")

    # Group by ryhat and calculate mean of y (support)
    support = df.groupby('ryhat')['y'].mean().reset_index(name='support')

    # Merge support back to the original dataframe
    df = pd.merge(df, support, on="ryhat", how="left")

    # Create the rord column by directly assigning 'support' where ryhat matches
    df['rord'] = df['support']

    # Sort df by yhat to get correct ordering for ystar
    df = df.sort_values(by="yhat").reset_index(drop=True)

    # Get ystar in the same order as rord in the sorted dataframe
    ystar = df['rord'].values

    # Create the index array i
    n = len(df)

    # y sorted in ascending order
    sorted_y = np.sort(df['y'].values)

    # Lorenz curve (y sorted ascending)
    lorenz_y = np.insert(np.cumsum(sorted_y) / (n * sorted_y.mean()), 0, 0)
    x_lorenz = np.insert(np.linspace(1 / n, 1, n), 0, 0)

    # Dual Lorenz curve (y sorted descending)
    dual_y = np.insert(np.cumsum(sorted_y[::-1]) / (n * sorted_y.mean()), 0, 0)
    x_dual = np.insert(np.linspace(1 / n, 1, n), 0, 0)

    # Concordance curve (ystar sorted by yhat)
    concord_y = np.insert(np.cumsum(ystar) / (n * sorted_y.mean()), 0, 0)
    x_concord = np.insert(np.linspace(1 / n, 1, n), 0, 0)

    total_denom = np.trapz(dual_y - lorenz_y, x_concord)

    # Slice
    mask = (x_concord >= lower) & (x_concord <= upper)
    x_slice = x_concord[mask]
    num_slice = np.trapz((dual_y[mask] - concord_y[mask]), x_slice)
    partial_rga_contribution = num_slice / total_denom

    # Local RGA
    denom_slice = np.trapz(dual_y[mask] - lorenz_y[mask], x_slice)
    rga_local = num_slice / denom_slice

    if plot:
        # Plot of curves
        plt.figure(figsize=(8, 6))
        plt.plot(x_lorenz, lorenz_y, label="Lorenz", color="yellow")
        plt.plot(x_dual, dual_y, label="Dual Lorenz", color="green")
        plt.plot(x_concord, concord_y, label="Concordance (C)", color="red")
        plt.plot([0, 1], [0, 1], '--', color='gray', label='45-degree line')
        plt.fill_between(x_slice, concord_y[mask], dual_y[mask], color="gray", alpha=0.3)
        plt.xlabel("Cumulative proportion p")
        plt.ylabel("Cumulative proportion f(p)")
        plt.title(f"Lorenz, Dual Lorenz, and Concordance Curves\nSegment [{lower:.2f}–{upper:.2f}]: "
                  f"Partial RGA = {partial_rga_contribution:.3f}, "
                  f"Local RGA = {rga_local:.3f}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return partial_rga_contribution