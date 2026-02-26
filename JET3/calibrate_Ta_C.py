"""
Calibrate Air Temperature (Ta_C) using OLS regression coefficients.

This module provides a function to correct raw/uncalibrated air temperature estimates
by predicting and subtracting their systematic error using OLS regression coefficients
derived from validation data.

The coefficients are stored externally as CSV and loaded at runtime.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def calibrate_Ta_C(data, raw_ta_c_col='Ta_C', output_col='Ta_C_cal'):
    """
    Calibrate air temperature estimates by applying OLS error correction.
    
    This function predicts the systematic error in raw Ta_C estimates and subtracts
    it to produce calibrated values. Uses only remote sensing inputs (no in-situ data).
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data with the following required columns:
        - raw_ta_c_col (default='Ta_C'): Column with raw temperature estimates
        - NDVI, ST_C, SZA_deg, albedo, canopy_height_meters, elevation_m, 
          emissivity, wind_speed_mps (predictor variables)
    raw_ta_c_col : str, optional
        Name of column containing raw Ta_C estimates (default: 'Ta_C')
    output_col : str, optional
        Name for output column with calibrated values (default: 'Ta_C_cal')
    
    Returns
    -------
    pd.DataFrame
        Copy of input data with added column:
        - output_col: Calibrated Ta_C values
        - '{raw_ta_c_col}_predicted_error': Predicted systematic error
    
    Examples
    --------
    >>> import pandas as pd
    >>> from JET3.calibrate_Ta_C import calibrate_Ta_C
    >>> 
    >>> # Load remote sensing data with raw estimates
    >>> data = pd.read_csv('remote_sensing_data.csv')
    >>> 
    >>> # Calibrate
    >>> calibrated = calibrate_Ta_C(data)
    >>> 
    >>> # Access results
    >>> print(calibrated[['Ta_C', 'Ta_C_cal']])
    
    Notes
    -----
    - Model Performance: R² = 0.2973, RMSE = 2.1664, MAE = 1.6223
    - Calibration formula: Ta_C_cal = Ta_C - predicted_error
    - Predictor variables must not contain NaN values
    - Coefficients were derived from ECOv002 cal/val dataset
    """
    # Load coefficients from CSV
    coef_path = Path(__file__).parent / "Ta_C_calibration_coefficients.csv"
    
    if not coef_path.exists():
        raise FileNotFoundError(
            f"Coefficient file not found: {coef_path}\n"
            "Please ensure Ta_C_calibration_coefficients.csv is in the JET3 package directory."
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
    
    # Check that raw Ta_C column exists
    if raw_ta_c_col not in data.columns:
        raise ValueError(f"Raw temperature column '{raw_ta_c_col}' not found in input data")
    
    # Check that all required predictor variables are in input data
    missing_vars = [v for v in predictor_vars if v not in data.columns]
    if missing_vars:
        raise ValueError(
            f"Missing required predictor variables in input data: {missing_vars}\n"
            f"Required variables: {predictor_vars}"
        )
    
    # Make a copy to avoid modifying original
    result = data.copy()
    
    # Extract predictor values
    X = data[predictor_vars].copy()
    
    # Check for NaN values
    has_nan = X.isna().any(axis=1)
    if has_nan.any():
        raise ValueError(
            f"Input data contains {has_nan.sum()} rows with NaN values in predictor variables.\n"
            "Please remove or impute missing values before calling this function."
        )
    
    # Apply OLS regression to predict systematic error
    # error = intercept + sum(coef_i * predictor_i)
    predicted_error = np.full(len(data), intercept, dtype=float)
    
    for _, row in predictor_coefs.iterrows():
        var = row['Variable']
        coef = row['Coefficient']
        predicted_error += coef * X[var].values
    
    # Store predicted error
    result[f'{raw_ta_c_col}_predicted_error'] = predicted_error
    
    # Apply calibration: calibrated = raw - predicted_error
    result[output_col] = data[raw_ta_c_col] - predicted_error
    
    return result
