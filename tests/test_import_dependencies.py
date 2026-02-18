import pytest

# List of dependencies
dependencies = [
    "AquaSEBS",
    "BESS_JPL",
    "check_distribution",
    "FLiESANN",
    "GEOS5FP",
    "MODISCI",
    "numpy",
    "onnxruntime",
    "pandas",
    "PMJPL",
    "PTJPL",
    "PTJPLSM",
    "dateutil",
    "pytictoc",
    "rasters",
    "SEBAL_soil_heat_flux",
    "shapely",
    "STIC_JPL",
    "verma_net_radiation"
]

# Generate individual test functions for each dependency
@pytest.mark.parametrize("dependency", dependencies)
def test_dependency_import(dependency):
    __import__(dependency)
