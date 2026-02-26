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


def calibrate_Ta_C(
    Ta_C: np.ndarray,
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
    Calibrate air temperature estimates by applying OLS error correction.
    
    This function predicts the systematic error in raw Ta_C estimates and subtracts
    it to produce calibrated values. Uses only remote sensing inputs (no in-situ data).
    
    Parameters
    ----------
    Ta_C : np.ndarray
        Air temperature estimates in Celsius to be calibrated
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
        Calibrated air temperature values (raw - predicted_error)
    
    Examples
    --------
    >>> import numpy as np
    >>> from JET3.calibrate_Ta_C import calibrate_Ta_C
    >>> 
    >>> # Example with 10 samples
    >>> Ta_C = np.array([25.5, 26.2, 27.1, ...])
    >>> NDVI = np.array([0.5, 0.6, 0.7, ...])
    >>> ST_C = np.array([35.2, 36.1, 37.5, ...])
    >>> # ... provide all 8 predictors
    >>> 
    >>> # Calibrate
    >>> calibrated = calibrate_Ta_C(Ta_C, NDVI, ST_C, SZA_deg, albedo, 
    ...                            canopy_height_meters, elevation_m, 
    ...                            emissivity, wind_speed_mps)
    
    Notes
    -----
    - Model Performance: R² = 0.2973, RMSE = 2.1664, MAE = 1.6223
    - Calibration formula: Ta_C_cal = Ta_C - predicted_error
    - All input arrays must have the same length
    - Input arrays may contain NaN values; output will be NaN at those positions
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
    
    Ta_C = np.asarray(Ta_C)
    
    # Check array lengths match
    n = len(Ta_C)
    for var_name, arr in predictors.items():
        if len(arr) != n:
            raise ValueError(
                f"Input array length mismatch: {var_name} has length {len(arr)}, "
                f"but Ta_C has length {n}"
            )
    
    # Create mask for valid (non-NaN) values across all inputs
    valid_mask = np.ones(n, dtype=bool)
    for arr in predictors.values():
        valid_mask &= ~np.isnan(arr)
    valid_mask &= ~np.isnan(Ta_C)
    
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
        calibrated[valid_mask] = Ta_C[valid_mask] - predicted_error
    
    return calibrated
