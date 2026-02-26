"""
Generate uncertainty quantification (UQ) for calibrated Soil Moisture (SM).

This module provides a function to estimate the ±1-sigma uncertainty of calibrated
soil moisture estimates using OLS regression coefficients derived from validation data.

The coefficients are stored externally as CSV and loaded at runtime.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_SM_calibrated_UQ(
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
    Generate ±1-sigma uncertainty quantification for calibrated soil moisture estimates.
    
    This function applies an OLS regression model trained on validation data to predict
    the expected absolute error (uncertainty) of calibrated SM estimates using
    only remote sensing inputs.
    
    Parameters
    ----------
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
        The ±1-sigma uncertainty magnitude for each input observation.
        Uncertainty values are guaranteed to be non-negative.
        
    Examples
    --------
    >>> import numpy as np
    >>> from JET3.generate_SM_calibrated_UQ import generate_SM_calibrated_UQ
    >>> 
    >>> # Example with 10 samples
    >>> NDVI = np.array([0.5, 0.6, 0.7, ...])
    >>> ST_C = np.array([35.2, 36.1, 37.5, ...])
    >>> # ... provide all 8 predictors
    >>> 
    >>> # Generate calibrated UQ
    >>> uq = generate_SM_calibrated_UQ(NDVI, ST_C, SZA_deg, albedo,
    ...                                 canopy_height_meters, elevation_m,
    ...                                 emissivity, wind_speed_mps)
    >>> 
    >>> # Use with calibrated estimates
    >>> calibrated_sm = np.array([0.24, 0.27, 0.30, ...])
    >>> lower_bound = calibrated_sm - uq
    >>> upper_bound = calibrated_sm + uq
    
    Notes
    -----
    - Model Performance: R² = 0.1757, RMSE = 0.0465, MAE = 0.0331
    - All input arrays must have the same length
    - Input arrays must not contain NaN values
    - Coefficients were derived from ECOv002 cal/val dataset
    - These coefficients predict uncertainty of calibrated values (after error correction)
    """
    # Load coefficients from CSV
    coef_path = Path(__file__).parent / "SM_calibrated_UQ_coefficients.csv"
    
    if not coef_path.exists():
        raise FileNotFoundError(
            f"Coefficient file not found: {coef_path}\n"
            "Please ensure SM_calibrated_UQ_coefficients.csv is in the JET3 package directory."
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
    
    # Check array lengths match
    n = None
    for var_name, arr in predictors.items():
        arr_len = len(arr)
        if n is None:
            n = arr_len
        elif len(arr) != n:
            raise ValueError(
                f"Input array length mismatch: {var_name} has length {arr_len}, "
                f"but other arrays have length {n}"
            )
    
    # Check for NaN values
    for var_name, arr in predictors.items():
        if np.isnan(arr).any():
            raise ValueError(
                f"Input array '{var_name}' contains NaN values.\n"
                "Please remove or impute missing values before calling this function."
            )
    
    # Apply OLS regression: UQ = intercept + sum(coef_i * predictor_i)
    uq = np.full(n, intercept, dtype=float)
    
    for _, row in predictor_coefs.iterrows():
        var = row['Variable']
        coef = row['Coefficient']
        if var not in predictors:
            raise ValueError(f"Predictor '{var}' from coefficients not found in input parameters")
        uq += coef * predictors[var]
    
    # Ensure non-negative uncertainty
    uq = np.maximum(uq, 0)
    
    return uq
