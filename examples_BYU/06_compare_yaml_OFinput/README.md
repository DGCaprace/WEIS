
# Notes

This examples aims to show that we can use AeroelasticSE through its interface with WEIS, instead of using it standalone.
The advantages are:
- WEIS takes care of the generation of all input files for OpenFAST, reading information from the turbine yaml file. Hence we are sure there is consistency between WISDEM runs and OpenFAST runs
- WEIS handles automatically the generation of IEC cases, that we would need to handle manually if we were to use standalone AeroelasticSE
- WEIS is maintained at NREL and will integrate all the latest updates and features (whereas we could miss some)
- Before running OpenFAST, WEIS takes care of retuning the ROSCO controller to the current turbine design, meaning that we always run OpenFAST with an up-to-date controller. This is done by running a power curve through WISDEM prior to any OpenFAST simulation, and using other information provided in the yaml. That would take substantial effort if we had to do that from our own script, YET the controller has a major influence on the unsteady loads so need to be implemented !!
  
The con's are:
- for simplicity and to limit the changes that we do to WEIS, we will need to implement the aggregation of DEL in WEIS directly (the other option would be to hack WEIS so that it outputs the time signals).
- WEIS was not designed to perform rotor-only simulations so needed to do some adjustments to allow it... and since the code is quite long, we can't be sure that rotor-only simulations are fully consistent (at least on WISDEM side; for OpenFAST, we actually made sure that the simulations are run with rigid tower).
  - In particular, we want rigid tower so we deactivate towerSE in the model. Consequently, DriverSE crashes because there is no tower so we need to turn that off too. In the end, I am not 100% sure that the tuning of the ROSCO controller can be done properly from WISDEM results obtained without DriverSE (Some assumptions are made on the drive parameters since DriverSE is deactivated).


# Running

Just run the driver.

The execution will throw many warnings related to the absence of tower.
Similarly, you will get
```
invalid value encountered in true_divideRuntimeWarning
```
in openmdao_openfast, just because WEIS tries to write NaNs for tower values in OpenFAST file. This is ok since we will deactivate the tower in ElastoDyn, such that we never read that file.