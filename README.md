# JET3: JPL Evapotranspiration Ensemble Algorithm

[![CI](https://github.com/JPL-Evapotranspiration-Algorithms/JET3/actions/workflows/ci.yml/badge.svg)](https://github.com/JPL-Evapotranspiration-Algorithms/JET3/actions/workflows/ci.yml)

JET3 is a Python package for computing evapotranspiration (ET), evaporative stress index (ESI), and water use efficiency (WUE) using an ensemble of four scientifically-validated models. This package provides the core algorithms used in the ECOSTRESS and Surface Biology and Geology (SBG) missions, now available as a standalone, mission-independent tool.

JET3 is part of the [JPL-Evapotranspiration-Algorithms](https://github.com/JPL-Evapotranspiration-Algorithms) organization and serves as the foundation for:
- [ECOSTRESS Collection 3 L3T/L4T JET Products](https://github.com/ECOSTRESS-Collection-3/ECOv003-L3T-L4T-JET)
- [SBG Collection 1 L4 JET Products](https://github.com/sbg-tir/SBG-TIR-L4-JET)

## Authors

[Gregory H. Halverson](https://github.com/gregory-halverson-jpl) (they/them)<br>
[gregory.h.halverson@jpl.nasa.gov](mailto:gregory.h.halverson@jpl.nasa.gov)<br>
NASA Jet Propulsion Laboratory 329G

Kerry Cawse-Nicholson (she/her)<br>
NASA Jet Propulsion Laboratory 329G

Madeleine Pascolini-Campbell (she/her)<br>
NASA Jet Propulsion Laboratory 329F

This package has been developed using open-science practices with New Technology Report (NTR) and open-source license from NASA Jet Propulsion Laboratory. The code is based on algorithms originally developed for the ECOSTRESS mission and refined for broader applications.

## 1. Introduction 

JET3 produces estimates of:
- **Evapotranspiration (ET)**: The combined process of water evaporation from soil and transpiration from plants
- **Evaporative Stress Index (ESI)**: A measure of vegetation water stress (ratio of actual to potential ET)
- **Water Use Efficiency (WUE)**: The ratio of carbon uptake (GPP) to water loss (ET)

Accurate modeling of ET requires consideration of many environmental and biological controls including: solar radiation, atmospheric water vapor deficit, soil water availability, vegetation physiology and phenology (Brutsaert, 1982; Monteith, 1965; Penman, 1948). 

JET3 implements an ensemble approach combining four scientifically-validated models:
- **PT-JPL-SM**: Priestley-Taylor model with soil moisture sensitivity
- **STIC-JPL**: Surface Temperature Initiated Closure model
- **PM-JPL**: Penman-Monteith model (MOD16 derivative)
- **BESS-JPL**: Breathing Earth System Simulator model

The ensemble median provides robust ET estimates with reduced uncertainty compared to individual models.

### Required Inputs

JET3 requires the following input variables:
- **Surface temperature (ST)**: Land surface temperature in Celsius
- **NDVI**: Normalized Difference Vegetation Index
- **Albedo**: Surface albedo
- **Air temperature (Ta)**: Near-surface air temperature in Celsius
- **Relative humidity (RH)**: Near-surface relative humidity (0-1)
- **Soil moisture (SM)**: Volumetric soil moisture (m³/m³)
- **Net radiation (Rn)**: Net radiation in W/m²
- **Additional meteorological variables** for individual models

These inputs can be derived from any appropriate remote sensing or ground-based data sources.

## 2. Installation

Install from PyPI:
```bash
pip install JET3
```

Or install from source:
```bash
git clone https://github.com/JPL-Evapotranspiration-Algorithms/JET3.git
cd JET3
pip install -e .
```

## 3. Usage

```python
from JET3 import JET

# Initialize JET ensemble processor
jet = JET()

# Compute ET ensemble with your input data
results = jet.process(
    ST=surface_temperature,      # Surface temperature (°C)
    NDVI=ndvi,                   # Normalized Difference Vegetation Index
    albedo=albedo,               # Surface albedo
    Ta=air_temperature,          # Air temperature (°C)
    RH=relative_humidity,        # Relative humidity (0-1)
    SM=soil_moisture,            # Soil moisture (m³/m³)
    Rn=net_radiation,            # Net radiation (W/m²)
    # ... other required inputs
)

# Access individual model outputs
et_ptjplsm = results['PTJPLSMinst']
et_stic = results['STICJPLdaily']
et_pmjpl = results['PMJPLdaily']
et_bess = results['BESSJPLdaily']

# Access ensemble estimate
et_ensemble = results['ETdaily']
et_uncertainty = results['ETinstUncertainty']

# Access derived products
esi = results['ESI']  # Evaporative Stress Index
wue = results['WUE']  # Water Use Efficiency
```

## 4. Evapotranspiration Models

JET3 implements an ensemble of four evapotranspiration models, each with different strengths and theoretical foundations. The ensemble approach combines outputs to reduce uncertainty and improve overall accuracy.

### 4.1. Priestley-Taylor (PT-JPL-SM) Evapotranspiration Model

The Priestley-Taylor Jet Propulsion Laboratory model with Soil Moisture (PT-JPL-SM), developed by Dr. Adam Purdy and Dr. Joshua Fisher, was designed as a soil moisture-sensitive evapotranspiration product for the Soil Moisture Active-Passive (SMAP) mission. The model estimates instantaneous canopy transpiration, leaf surface evaporation, and soil moisture evaporation using the Priestley-Taylor formula with a set of constraints. These three partitions are combined into total latent heat flux in watts per square meter for the ensemble estimate.

**Reference**: Purdy, A.J., Fisher, J.B., Goulden, M.L., Colliander, A., Halverson, G., Tu, K., Famiglietti, J.S. (2018). SMAP soil moisture improves global evapotranspiration. *Remote Sensing of Environment*, 219, 1-14. https://doi.org/10.1016/j.rse.2018.09.023

**Repository**: [PT-JPL-SM](https://github.com/JPL-Evapotranspiration-Algorithms/PT-JPL-SM)

### 4.2. Surface Temperature Initiated Closure (STIC-JPL) Evapotranspiration Model

The Surface Temperature Initiated Closure-Jet Propulsion Laboratory (STIC-JPL) model, contributed by Dr. Kaniska Mallick, was designed as a surface temperature-sensitive ET model, adopted by ECOSTRESS and SBG for improved estimates of ET reflecting mid-day heat stress. The STIC-JPL model estimates total latent heat flux directly using thermal remote sensing observations. This instantaneous estimate of latent heat flux is included in the ensemble estimate.

**Reference**: Mallick, K., Trebs, I., Boegh, E., Giustarini, L., Schlerf, M., Drewry, D.T., Hoffmann, L., von Randow, C., Kruijt, B., Araùjo, A., Saleska, S., Ehleringer, J.R., Domingues, T.F., Ometto, J.P.H.B., Nobre, A.D., de Moraes, O.L.L., Hayek, M., Munger, J.W., Wofsy, S.C. (2016). Canopy-scale biophysical controls of transpiration and evaporation in the Amazon Basin. *Hydrology and Earth System Sciences*, 20, 4237-4264. https://doi.org/10.5194/hess-20-4237-2016

**Repository**: [STIC-JPL](https://github.com/JPL-Evapotranspiration-Algorithms/STIC-JPL)

### 4.3. Penman Monteith (PM-JPL) Evapotranspiration Model

The Penman-Monteith-Jet Propulsion Laboratory (PM-JPL) algorithm is a derivation of the MOD16 algorithm that was originally designed as the ET product for the Moderate Resolution Imaging Spectroradiometer (MODIS) and continued as a Visible Infrared Imaging Radiometer Suite (VIIRS) product. PM-JPL uses a similar approach to PT-JPL and PT-JPL-SM to independently estimate vegetation and soil components of instantaneous ET, but using the Penman-Monteith formula instead of the Priestley-Taylor. The PM-JPL latent heat flux partitions are summed to total latent heat flux for the ensemble estimate.

**Reference**: Running, S., Mu, Q., Zhao, M., Moreno, A. (2019). MODIS Global Terrestrial Evapotranspiration (ET) Product (MOD16A2/A3 and MOD16A2GF/A3GF). NASA Earth Observing System Data and Information System (EOSDIS) Land Processes Distributed Active Archive Center (LP DAAC). https://doi.org/10.5067/MODIS/MOD16A2.061

**Repository**: [PM-JPL](https://github.com/JPL-Evapotranspiration-Algorithms/PM-JPL)

### 4.4. Breathing Earth System Simulator (BESS-JPL) Gross Primary Production (GPP) Model

The Breathing Earth System Simulator Jet Propulsion Laboratory (BESS-JPL) model is a coupled surface energy balance and photosynthesis model contributed by Dr. Youngryel Ryu. The model iteratively calculates net radiation, ET, and Gross Primary Production (GPP) estimates. The latent heat flux component of BESS-JPL is included in the ensemble estimate, while the BESS-JPL net radiation is used as input to the other ET models.

**Reference**: Ryu, Y., Baldocchi, D.D., Kobayashi, H., van Ingen, C., Li, J., Black, T.A., Beringer, J., van Gorsel, E., Knohl, A., Law, B.E., Roupsard, O. (2011). Integration of MODIS land and atmosphere products with a coupled-process model to estimate gross primary productivity and evapotranspiration from 1 km to global scales. *Global Biogeochemical Cycles*, 25, GB4017. https://doi.org/10.1029/2011GB004053

**Repository**: [BESS-JPL](https://github.com/JPL-Evapotranspiration-Algorithms/BESS-JPL)

### 4.5. Ensemble Processing

The ensemble ET estimate is computed as the median of total latent heat flux (in watts per square meter) from the PT-JPL-SM, STIC-JPL, PM-JPL, and BESS-JPL models. This median is then upscaled to a daily ET estimate in millimeters per day. The standard deviation between these multiple estimates represents the ensemble uncertainty.

### 4.6. AquaSEBS Water Surface Evaporation

For water surface pixels, JET3 implements the AquaSEBS (Aquatic Surface Energy Balance System) model developed by Abdelrady et al. (2016) and validated by Fisher et al. (2023). Water surface evaporation is calculated using a physics-based approach that combines the equilibrium temperature model for water heat flux with the Priestley-Taylor equation for evaporation estimation.

**References**: 
- Abdelrady, A., Timmermans, J., Vekerdy, Z., Salama, M.S. (2016). Surface Energy Balance of Fresh and Saline Waters: AquaSEBS. *Remote Sensing*, 8, 583. https://doi.org/10.3390/rs8070583
- Fisher, J.B., Dohlen, M.B., Halverson, G.H., Collison, J.W., Hook, S.J., Hulley, G.C. (2023). Remotely sensed terrestrial open water evaporation. *Scientific Reports*, 13, 8217. https://doi.org/10.1038/s41598-023-34921-2

**Repository**: [AquaSEBS](https://github.com/JPL-Evapotranspiration-Algorithms/AquaSEBS)

#### Methodology

The AquaSEBS model implements the surface energy balance equation specifically adapted for water bodies:

$$R_n = LE + H + W$$

Where the water heat flux (W) is calculated using the equilibrium temperature model:

$$W = \beta \times (T_e - WST)$$

The key parameters include:
- **Temperature difference**: $T_n = 0.5 \times (WST - T_d)$ where WST is water surface temperature and $T_d$ is dew point temperature
- **Evaporation efficiency**: $\eta = 0.35 + 0.015 \times WST + 0.0012 \times T_n^2$
- **Thermal exchange coefficient**: $\beta = 4.5 + 0.05 \times WST + (\eta + 0.47) \times S$
- **Equilibrium temperature**: $T_e = T_d + \frac{SW_{net}}{\beta}$

Latent heat flux is then calculated using the Priestley-Taylor equation with α = 1.26 for water surfaces:

$$LE = \alpha \times \frac{\Delta}{\Delta + \gamma} \times (R_n - W)$$

#### Validation and Accuracy

The AquaSEBS methodology has been extensively validated against 19 in situ open water evaporation sites worldwide spanning multiple climate zones. Performance metrics include:

**Daily evaporation estimates:**
- **MODIS-based**: r² = 0.47, RMSE = 1.5 mm/day (41% of mean), Bias = 0.19 mm/day
- **Landsat-based**: r² = 0.56, RMSE = 1.2 mm/day (38% of mean), Bias = -0.8 mm/day

**Instantaneous estimates (controlled for high wind events >7.5 m/s):**
- **Correlation**: r² = 0.71
- **RMSE**: 53.7 W/m² (38% of mean)
- **Bias**: -19.1 W/m² (13% of mean)

The model demonstrates particular strength in water-limited environments and performs well across spatial scales from 30m (Landsat) to 1km (MODIS) resolution.

Water surface evaporation estimates are included in the `ETdaily` layer in mm per day, integrated over the daylight period from sunrise to sunset.

### 4.7. Evaporative Stress Index (ESI) and Water Use Efficiency (WUE)

The PT-JPL-SM model generates estimates of both actual and potential instantaneous ET. The potential evapotranspiration (PET) estimate represents the maximum expected ET if there were no water stress to plants on the ground. The ratio of the actual ET estimate to the PET estimate forms an index representing the water stress of plants, with zero being fully stressed with no observable ET and one being non-stressed with ET reaching PET.

**Evaporative Stress Index (ESI)**: 
$$\text{ESI} = \frac{\text{ET}_{\text{actual}}}{\text{PET}}$$

Water Use Efficiency (WUE) relates the amount of carbon that plants are taking in (GPP from BESS-JPL) to the amount of water that plants are releasing (transpiration from PT-JPL-SM):

**Water Use Efficiency (WUE)**:
$$\text{WUE} = \frac{\text{GPP}}{\text{Transpiration}}$$

WUE is expressed as the ratio of grams of carbon that plants take in to kilograms of water that plants release ($\text{g C kg}^{-1} \text{H}_2\text{O}$).

## 5. Theory

The JPL evapotranspiration (JET) ensemble provides a robust estimation of ET from multiple ET models. The ET ensemble incorporates ET data from four algorithms: Priestley Taylor-Jet Propulsion Laboratory model with soil moisture (PT-JPL-SM), the Penman Monteith-Jet Propulsion Laboratory model (PM-JPL), Surface Temperature Initiated Closure-Jet Propulsion Laboratory model (STIC-JPL), and the Breathing Earth System Simulator-Jet Propulsion Laboratory model (BESS-JPL). 

Each model brings complementary strengths:
- **PT-JPL-SM**: Incorporates soil moisture constraints and partitions ET into canopy transpiration, interception, and soil evaporation
- **STIC-JPL**: Leverages surface temperature to detect heat stress and reduced ET capacity
- **PM-JPL**: Implements the widely-used Penman-Monteith formulation with aerodynamic considerations
- **BESS-JPL**: Couples photosynthesis with ET through a comprehensive surface energy balance

The ensemble median approach reduces model-specific biases and provides more robust estimates than any individual model.

## 6. Validation

The JET ensemble approach has been validated against flux tower measurements from the FLUXNET network as documented in Pierrat et al. (2025) and through the ECOSTRESS mission. The validation demonstrated that the ensemble evapotranspiration estimates:

- Show strong correlation with flux tower measurements (R² > 0.7) across most biomes
- Capture the diurnal and seasonal patterns of evapotranspiration effectively
- Perform well in water-limited ecosystems where thermal stress indicators are most valuable
- Benefit from the ensemble approach, with the median estimate generally outperforming individual models
- Maintain accuracy across a range of spatial scales

### Performance by Biome

The validation results indicate varying performance across different ecosystem types:

- **Croplands**: Excellent agreement during growing season, capturing irrigation and phenological patterns
- **Forests**: Good performance in temperate and boreal forests, with some challenges in dense tropical canopies
- **Grasslands**: Strong performance in both natural and managed grassland systems
- **Shrublands**: Reliable estimates in semi-arid regions where thermal stress is prevalent

The JET3 ensemble provides reliable ET estimates suitable for water resource management, agricultural monitoring, and ecosystem research applications.

## 7. Acknowledgements 

We would like to thank Joshua Fisher as the initial science lead of the ECOSTRESS mission and PI of the ROSES project to develop the JET ensemble approach.

We would like to thank Adam Purdy for developing the PT-JPL-SM model.

We would like to thank Kaniska Mallick for contributing the STIC model.

We would like to thank Youngryel Ryu for contributing the BESS-JPL model.

## 8. References

- Abdelrady, A., Timmermans, J., Vekerdy, Z., Salama, M.S. (2016). Surface Energy Balance of Fresh and Saline Waters: AquaSEBS. *Remote Sensing*, 8, 583. https://doi.org/10.3390/rs8070583
- Allen, R.G., Tasumi, M., & Trezza, R. (2007). "Satellite-based energy balance for mapping evapotranspiration with internalized calibration (METRIC)—Model." *Journal of Irrigation and Drainage Engineering*, 133(4), 380-394. https://doi.org/10.1061/(ASCE)0733-9437(2007)133:4(380)
- Brutsaert, W. (1982). *Evaporation into the Atmosphere: Theory, History, and Applications*. Springer Netherlands. https://doi.org/10.1007/978-94-017-1497-6
- Fisher, J.B., Dohlen, M.B., Halverson, G.H., Collison, J.W., Hook, S.J., Hulley, G.C. (2023). Remotely sensed terrestrial open water evaporation. *Scientific Reports*, 13, 8217. https://doi.org/10.1038/s41598-023-34921-2
- Fisher, J.B., Tu, K.P., Baldocchi, D.D. (2008). Global estimates of the land–atmosphere water flux based on monthly AVHRR and ISLSCP-II data, validated at 16 FLUXNET sites. *Remote Sensing of Environment*, 112(3), 901-919. https://doi.org/10.1016/j.rse.2007.06.025
- Mallick, K., Trebs, I., Boegh, E., Giustarini, L., Schlerf, M., Drewry, D.T., Hoffmann, L., von Randow, C., Kruijt, B., Araùjo, A., Saleska, S., Ehleringer, J.R., Domingues, T.F., Ometto, J.P.H.B., Nobre, A.D., de Moraes, O.L.L., Hayek, M., Munger, J.W., Wofsy, S.C. (2016). Canopy-scale biophysical controls of transpiration and evaporation in the Amazon Basin. *Hydrology and Earth System Sciences*, 20, 4237-4264. https://doi.org/10.5194/hess-20-4237-2016
- Monteith, J.L. (1965). "Evaporation and Environment." *Symposia of the Society for Experimental Biology*, 19, 205-234.
- Penman, H.L. (1948). "Natural Evaporation from Open Water, Bare Soil and Grass." *Proceedings of the Royal Society of London. Series A, Mathematical and Physical Sciences*, 193(1032), 120-145. https://doi.org/10.1098/rspa.1948.0037
- Pierrat, Z., et al. (2025). "Validation of ECOSTRESS Collection 3 Evapotranspiration Products Using FLUXNET Measurements." *Remote Sensing of Environment* (in press).
- Purdy, A.J., Fisher, J.B., Goulden, M.L., Colliander, A., Halverson, G., Tu, K., Famiglietti, J.S. (2018). SMAP soil moisture improves global evapotranspiration. *Remote Sensing of Environment*, 219, 1-14. https://doi.org/10.1016/j.rse.2018.09.023
- Running, S., Mu, Q., Zhao, M., Moreno, A. (2019). MODIS Global Terrestrial Evapotranspiration (ET) Product (MOD16A2/A3 and MOD16A2GF/A3GF). NASA Earth Observing System Data and Information System (EOSDIS) Land Processes Distributed Active Archive Center (LP DAAC). https://doi.org/10.5067/MODIS/MOD16A2.061
- Ryu, Y., Baldocchi, D.D., Kobayashi, H., van Ingen, C., Li, J., Black, T.A., Beringer, J., van Gorsel, E., Knohl, A., Law, B.E., Roupsard, O. (2011). Integration of MODIS land and atmosphere products with a coupled-process model to estimate gross primary productivity and evapotranspiration from 1 km to global scales. *Global Biogeochemical Cycles*, 25, GB4017. https://doi.org/10.1029/2011GB004053
