# PCCC-MOF-Adsorption-Model
This is a machine learning model to predict adsorption properties (CO2 working capacity and CO2/N2 selectivity) of MOFs under post-combustion carbon capture conditions. The model was trained on the ARC-MOF database. The code uses two MLP models with three hidden layers each to predict working capacity and selectivity, using geometric and AP-RDF descriptors.

Required Python libraries (tested versions):
-torch (version 1.9.0)
-PyCifRW (version 4.4.3)

Also required: Zeo++ (installation instructions: http://www.zeoplusplus.org/download.html)

Can run the code with the --help option for usage instructions:
usage: predict.py [-h] [-discard_geo_props DISCARD_GEO_PROPS] cif zeo_exe

Predict working capacity for a MOF under post-combustion carbon capture conditions

positional arguments:
  cif                   CIF file of the structure
  zeo_exe               Path of the zeo++ executable

optional arguments:
  -h, --help            show this help message and exit
  -discard_geo_props DISCARD_GEO_PROPS
                        Whether to discard the geometric properties. If false, they are written in the present directory.

