"""
Generate uncertainty quantification (UQ) for uncalibrated Relative Humidity (RH).

This module provides a function to estimate the ±1-sigma uncertainty of raw/uncalibrated
relative humidity estimates using OLS regression coefficients derived from validation data.

The coefficients are stored externally as CSV and loaded at runtime.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_RH_uncalibrated_UQ(ndvi, st_c, sza_deg, albedo, canopy_height_meters, 
                                elevation_m, emissivity, wind_speed_mps):
    """
    Generate ±1-sigma uncertainty quantification for uncalibrated relative humidity estimates.
    
    This function applies an OLS regression model trained on validation data to predict
    the expected absolute error (uncertainty) of raw/uncalibrated RH estimates using
    only remote sensing inputs.
    
    Parameters
    ----------
    ndvi : np.ndarray
        Normalized Difference Vegetation Index
    st_c : np.ndarray
        Surface Temperature in Celsius
    sza_deg : np.ndarray
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
    >>> from JET3.generate_RH_uncalibrated_UQ import generate_RH_uncalibrated_UQ
    >>> 
    >>> # Example with 10 samples
    >>> ndvi = np.array([0.5, 0.6, 0.7, ...])
    >>> st_c = np.array([35.2, 36.1, 37.5, ...])
    >>> # ... provide all 8 predictors
    >>> 
    >>> # Generate UQ
    >>> uq = generate_RH_uncalibrated_UQ(ndvi, st_c, sza_deg, albedo,
    ...                                   canopy_height_meters, elevation_m,
    ...                                   emissivity, wind_speed_mps)
    >>> 
    >>> # Use with raw estimates
    >>> raw_rh = np.array([65.3, 66.1, 67.2, ...])
    >>> lower_bound = raw_rh - uq
    >>> upper_bound = raw_rh + uq
    
    Notes
    -----
    - Model Performance: R² = 0.2539, RMSE = 0.0707, MAE = 0.0549
    - All input arrays must have the same length
    - Input arrays must not contain NaN values
    - Coefficients were derived from ECOv002 cal/val dataset
    """
    # Load coefficients from CSV
    coef_path = Path(__file__).parent / "RH_uncalibrated_UQ_coefficients.csv"
    
    if not coef_path.exists():
        raise FileNotFoundError(
            f"Coefficient file not found: {coef_path}\n"
            "Please ensure RH_uncalibrated_UQ_coefficients.csv is in the JET3 package directory."
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
        'NDVI': np.asarray(ndvi),
        'ST_C': np.asarray(st_c),
        'SZA_deg': np.asarray(sza_deg),
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
