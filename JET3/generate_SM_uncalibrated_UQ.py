"""
Generate uncertainty quantification (UQ) for uncalibrated Soil Moisture (SM).

This module provides a function to estimate the ±1-sigma uncertainty of raw/uncalibrated
soil moisture estimates using OLS regression coefficients derived from validation data.

The coefficients are stored externally as CSV and loaded at runtime.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_SM_uncalibrated_UQ(data):
    """
    Generate ±1-sigma uncertainty quantification for uncalibrated soil moisture estimates.
    
    This function applies an OLS regression model trained on validation data to predict
    the expected absolute error (uncertainty) of raw/uncalibrated SM estimates using
    only remote sensing inputs.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data with the following required columns (predictor variables):
        - NDVI
        - ST_C
        - SZA_deg
        - albedo
        - canopy_height_meters
        - elevation_m
        - emissivity
        - wind_speed_mps
    
    Returns
    -------
    pd.Series
        The ±1-sigma uncertainty magnitude for each row in the input data.
        Uncertainty values are guaranteed to be non-negative.
        
    Examples
    --------
    >>> import pandas as pd
    >>> from JET3.generate_SM_uncalibrated_UQ import generate_SM_uncalibrated_UQ
    >>> 
    >>> # Load your remote sensing data
    >>> data = pd.read_csv('remote_sensing_data.csv')
    >>> 
    >>> # Generate UQ
    >>> uq = generate_SM_uncalibrated_UQ(data)
    >>> 
    >>> # Use with raw estimates
    >>> raw_sm = data['SM']
    >>> lower_bound = raw_sm - uq
    >>> upper_bound = raw_sm + uq
    
    Notes
    -----
    - Model Performance: R² = 0.1275, RMSE = 0.0619, MAE = 0.0434
    - Predictor variables must not contain NaN values
    - Coefficients were derived from ECOv002 cal/val dataset
    """
    # Load coefficients from CSV
    coef_path = Path(__file__).parent / "SM_uncalibrated_UQ_coefficients.csv"
    
    if not coef_path.exists():
        raise FileNotFoundError(
            f"Coefficient file not found: {coef_path}\n"
            "Please ensure SM_uncalibrated_UQ_coefficients.csv is in the JET3 package directory."
        )
    
    coef_df = pd.read_csv(coef_path)
    
    # Extract intercept
    intercept_row = coef_df[coef_df['Variable'] == 'Intercept']
    if intercept_row.empty:
        raise ValueError("Intercept coefficient not found in coefficient file")
    intercept = intercept_row['Coefficient'].values[0]
    
    # Extract coefficients for predictor variables
    predictor_coefs = coef_df[coef_df['Variable'] != 'Intercept'].copy()
    predictor_vars = predictor_coefs['Variable'].tolist()
    
    # Check that all required variables are in input data
    missing_vars = [v for v in predictor_vars if v not in data.columns]
    if missing_vars:
        raise ValueError(
            f"Missing required predictor variables in input data: {missing_vars}\n"
            f"Required variables: {predictor_vars}"
        )
    
    # Extract predictor values, handling NaN
    X = data[predictor_vars].copy()
    
    # Check for NaN values
    has_nan = X.isna().any(axis=1)
    if has_nan.any():
        raise ValueError(
            f"Input data contains {has_nan.sum()} rows with NaN values in predictor variables.\n"
            "Please remove or impute missing values before calling this function."
        )
    
    # Apply OLS regression: UQ = intercept + sum(coef_i * predictor_i)
    uq = np.full(len(data), intercept, dtype=float)
    
    for _, row in predictor_coefs.iterrows():
        var = row['Variable']
        coef = row['Coefficient']
        uq += coef * X[var].values
    
    # Ensure non-negative uncertainty
    uq = np.maximum(uq, 0)
    
    return pd.Series(uq, index=data.index, name='SM_uncalibrated_UQ')
