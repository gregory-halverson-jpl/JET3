"""
Calibrate Soil Moisture (SM) using OLS regression coefficients.

This module provides a function to correct raw/uncalibrated soil moisture estimates
by predicting and subtracting their systematic error using OLS regression coefficients
derived from validation data.

The coefficients are stored externally as CSV and loaded at runtime.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def calibrate_SM(data, raw_sm_col='SM', output_col='SM_cal'):
    """
    Calibrate soil moisture estimates by applying OLS error correction.
    
    This function predicts the systematic error in raw SM estimates and subtracts
    it to produce calibrated values. Uses only remote sensing inputs (no in-situ data).
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data with the following required columns:
        - raw_sm_col (default='SM'): Column with raw soil moisture estimates
        - NDVI, ST_C, SZA_deg, albedo, canopy_height_meters, elevation_m, 
          emissivity, wind_speed_mps (predictor variables)
    raw_sm_col : str, optional
        Name of column containing raw SM estimates (default: 'SM')
    output_col : str, optional
        Name for output column with calibrated values (default: 'SM_cal')
    
    Returns
    -------
    pd.DataFrame
        Copy of input data with added columns:
        - output_col: Calibrated SM values
        - '{raw_sm_col}_predicted_error': Predicted systematic error
    
    Examples
    --------
    >>> import pandas as pd
    >>> from JET3.calibrate_SM import calibrate_SM
    >>> 
    >>> # Load remote sensing data with raw estimates
    >>> data = pd.read_csv('remote_sensing_data.csv')
    >>> 
    >>> # Calibrate
    >>> calibrated = calibrate_SM(data)
    >>> 
    >>> # Access results
    >>> print(calibrated[['SM', 'SM_cal']])
    
    Notes
    -----
    - Model Performance: R² = 0.0577, RMSE = 0.0521, MAE = 0.0398
    - Calibration formula: SM_cal = SM - predicted_error
    - Predictor variables must not contain NaN values
    - Coefficients were derived from ECOv002 cal/val dataset
    """
    # Load coefficients from CSV
    coef_path = Path(__file__).parent / "SM_calibration_coefficients.csv"
    
    if not coef_path.exists():
        raise FileNotFoundError(
            f"Coefficient file not found: {coef_path}\n"
            "Please ensure SM_calibration_coefficients.csv is in the JET3 package directory."
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
    
    # Check that raw SM column exists
    if raw_sm_col not in data.columns:
        raise ValueError(f"Raw soil moisture column '{raw_sm_col}' not found in input data")
    
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
    result[f'{raw_sm_col}_predicted_error'] = predicted_error
    
    # Apply calibration: calibrated = raw - predicted_error
    result[output_col] = data[raw_sm_col] - predicted_error
    
    return result
