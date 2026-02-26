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


def calibrate_SM(
    SM: np.ndarray,
    NDVI: np.ndarray,
    ST_C: np.ndarray,
    SZA_deg: np.ndarray,
    albedo: np.ndarray,
    canopy_height_meters: np.ndarray,
    elevation_m: np.ndarray,
    emissivity: np.ndarray,
    wind_speed_mps: np.ndarray,
) -> np.ndarray:
    """
    Calibrate soil moisture estimates by applying OLS error correction.
    
    This function predicts the systematic error in raw SM estimates and subtracts
    it to produce calibrated values. Uses only remote sensing inputs (no in-situ data).
    
    Parameters
    ----------
    SM : np.ndarray
        Soil moisture estimates to be calibrated
    NDVI : np.ndarray
        Normalized Difference Vegetation Index
    ST_C : np.ndarray
        Surface Temperature in Celsius
    SZA_deg : np.ndarray
        Solar Zenith Angle in degrees
    albedo : np.ndarray
        Surface albedo
    canopy_height_meters : np.ndarray
        Canopy height in meters
    elevation_m : np.ndarray
        Elevation in meters
    emissivity : np.ndarray
        Surface emissivity
    wind_speed_mps : np.ndarray
        Wind speed in meters per second
    
    Returns
    -------
    np.ndarray
        Calibrated soil moisture values (raw - predicted_error)
    
    Examples
    --------
    >>> import numpy as np
    >>> from JET3.calibrate_SM import calibrate_SM
    >>> 
    >>> # Example with 10 samples
    >>> SM = np.array([0.2, 0.25, 0.3, ...])
    >>> NDVI = np.array([0.5, 0.6, 0.7, ...])
    >>> ST_C = np.array([35.2, 36.1, 37.5, ...])
    >>> # ... provide all 8 predictors
    >>> 
    >>> # Calibrate
    >>> calibrated = calibrate_SM(SM, NDVI, ST_C, SZA_deg, albedo, 
    ...                           canopy_height_meters, elevation_m, 
    ...                           emissivity, wind_speed_mps)
    
    Notes
    -----
    - Model Performance: R² = 0.0577, RMSE = 0.0521, MAE = 0.0398
    - Calibration formula: SM_cal = SM - predicted_error
    - All input arrays must have the same length
    - Input arrays may contain NaN values; output will be NaN at those positions
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
    
    # Build predictor dictionary
    predictors = {
        'NDVI': np.asarray(NDVI),
        'ST_C': np.asarray(ST_C),
        'SZA_deg': np.asarray(SZA_deg),
        'albedo': np.asarray(albedo),
        'canopy_height_meters': np.asarray(canopy_height_meters),
        'elevation_m': np.asarray(elevation_m),
        'emissivity': np.asarray(emissivity),
        'wind_speed_mps': np.asarray(wind_speed_mps),
    }
    
    SM = np.asarray(SM)
    
    # Check array lengths match
    n = len(SM)
    for var_name, arr in predictors.items():
        if len(arr) != n:
            raise ValueError(
                f"Input array length mismatch: {var_name} has length {len(arr)}, "
                f"but SM has length {n}"
            )
    
    # Create mask for valid (non-NaN) values across all inputs
    valid_mask = np.ones(n, dtype=bool)
    for arr in predictors.values():
        valid_mask &= ~np.isnan(arr)
    valid_mask &= ~np.isnan(SM)
    
    # Initialize output with NaN
    calibrated = np.full(n, np.nan, dtype=float)
    
    # Only calculate for valid positions
    if valid_mask.any():
        # Apply OLS regression to predict systematic error
        # error = intercept + sum(coef_i * predictor_i)
        predicted_error = np.full(valid_mask.sum(), intercept, dtype=float)
        
        for _, row in predictor_coefs.iterrows():
            var = row['Variable']
            coef = row['Coefficient']
            if var not in predictors:
                raise ValueError(f"Predictor '{var}' from coefficients not found in input parameters")
            predicted_error += coef * predictors[var][valid_mask]
        
        # Apply calibration: calibrated = raw - predicted_error
        calibrated[valid_mask] = SM[valid_mask] - predicted_error
    
    return calibrated
