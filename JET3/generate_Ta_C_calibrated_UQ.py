"""
Generate uncertainty quantification (UQ) for calibrated Air Temperature (Ta_C).

This module provides a function to estimate the ±1-sigma uncertainty of calibrated
air temperature estimates using OLS regression coefficients derived from validation data.

The coefficients are stored externally as CSV and loaded at runtime.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_Ta_C_calibrated_UQ(
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
    Generate ±1-sigma uncertainty quantification for calibrated air temperature estimates.
    
    This function applies an OLS regression model trained on validation data to predict
    the expected absolute error (uncertainty) of calibrated Ta_C estimates using
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
    >>> from JET3.generate_Ta_C_calibrated_UQ import generate_Ta_C_calibrated_UQ
    >>> 
    >>> # Example with 10 samples
    >>> NDVI = np.array([0.5, 0.6, 0.7, ...])
    >>> ST_C = np.array([35.2, 36.1, 37.5, ...])
    >>> # ... provide all 8 predictors
    >>> 
    >>> # Generate calibrated UQ
    >>> uq = generate_Ta_C_calibrated_UQ(NDVI, ST_C, SZA_deg, albedo,
    ...                                   canopy_height_meters, elevation_m,
    ...                                   emissivity, wind_speed_mps)
    >>> 
    >>> # Use with calibrated estimates
    >>> calibrated_ta_c = np.array([24.5, 25.2, 26.1, ...])
    >>> lower_bound = calibrated_ta_c - uq
    >>> upper_bound = calibrated_ta_c + uq
    
    Notes
    -----
    - Model Performance: R² = 0.0789, RMSE = 1.3780, MAE = 1.0622
    - All input arrays must have the same length
    - Input arrays may contain NaN values; output will be NaN at those positions
    - Coefficients were derived from ECOv002 cal/val dataset
    - These coefficients predict uncertainty of calibrated values (after error correction)
    """
    # Load coefficients from CSV
    coef_path = Path(__file__).parent / "Ta_C_calibrated_UQ_coefficients.csv"
    
    if not coef_path.exists():
        raise FileNotFoundError(
            f"Coefficient file not found: {coef_path}\n"
            "Please ensure Ta_C_calibrated_UQ_coefficients.csv is in the JET3 package directory."
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
    
    # Create mask for valid (non-NaN) values across all inputs
    valid_mask = np.ones(n, dtype=bool)
    for arr in predictors.values():
        valid_mask &= ~np.isnan(arr)
    
    # Initialize output with NaN
    uq = np.full(n, np.nan, dtype=float)
    
    # Only calculate for valid positions
    if valid_mask.any():
        # Apply OLS regression: UQ = intercept + sum(coef_i * predictor_i)
        uq_valid = np.full(valid_mask.sum(), intercept, dtype=float)
        
        for _, row in predictor_coefs.iterrows():
            var = row['Variable']
            coef = row['Coefficient']
            if var not in predictors:
                raise ValueError(f"Predictor '{var}' from coefficients not found in input parameters")
            uq_valid += coef * predictors[var][valid_mask]
        
        # Ensure non-negative uncertainty
        uq_valid = np.maximum(uq_valid, 0)
        
        # Assign to output
        uq[valid_mask] = uq_valid
    
    return uq
