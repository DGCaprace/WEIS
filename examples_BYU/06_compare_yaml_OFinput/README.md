
# Notes

This example is made to compare the original OpenFAST files we derived for the DTU10MW Madsen variant, and the same information gathered in a single yaml file.

We run WEIS to get an openfast model from the yaml file, and compare to simple runs of the original input files passed to another instance of AeroelasticSE.

Note: The simulation with WEIS has a ROSCO controller, while the original model works at constant RPM. Simulations of the original model are done with turbulent IEC, whereas WEIS runs constant velocity inflow.