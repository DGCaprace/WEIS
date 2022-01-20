
# Modifications for combined fidelity approach

Hereafter, we explain how the scripts of the various subfolder can be used. The last two sections propose full workflows to perform 
1. WISDEM optimizations with loads from OpenFAST
2. a combined fidelity approach optimization (in combination with an external high-fidelity framework)

Notes:
- All examples are based on a turbine model inspired from the DTU 10MW reference turbine. The most up-to-date version of the OpenFAST model is available in the `Madsen2019_model_BD` folder. A WindIO yaml file that correponds to that model is also used throughout the examples (yet with some variations): look for `Madsen2019_10_forWEIS.yaml`.
- This code should use Openmdao 3.9 (crashes with 3.14).


Contextual information, and examples of applications can be found in:
```
@inproceedings{Caprace:2022,
	author = {Denis-Gabriel Caprace and Adam Cardoza and Teagan Nakamoto and Andrew Ning and Marco Mangano and Joaquim R. R. A. Martins},
	booktitle = {AIAA Scitech Forum},
	title = {Incorporating High-Fidelity Aerostructural Analyses in Wind Turbine Rotor Optimization},
	year = {2022}}
```
Please cite us as appropriate.

## Case 4 - computeDELs

Early code to perform DEL simulations with AeroelasticSE (not WEIS).

## Case 5 - test_pass_DEL 

Early/deprecated code to generate damage equivalent loads (DEL).

## Case 6 - compare_yaml_OFinput

Script to compare simulation outputs between a turbine defined by a yaml file, and a turbine defined with a full openfast model (all the model files).

See Readme in that file.  

## Case 7 - itarated DEL computation, and/or WISDEM optimization

- `driver.py`  enables the computation of extreme and/or fatigue loads from a yaml file. Procede to load extrapolation and aggregation. Outputs the loads in yaml files. Optionally run a subsequent WISDEM optimization. Optionally iterate over that process.  See top of that file for user definitions.

- `plot_results` is a sample plot script that can be adapted to plot various things from WISDEM output.

- `plot_results_HiFi` gathers results from several optimziation an plot them (SPECIFIC TO COMBINED FIDELITY APPROACH)

- `plot_EXTR_extrap`  can be used to do nicer plots of the load extrapolation procedure from the driver. Works on a `.npz` file output by the driver to bypass the hard load processing.

- `plot_EXTR_separately`  takes the result of various DLC computations (i.e. the analysis_XXX.yaml) and creates a map of the loadings experienced in the corresponding simulations. 

See Readme in that file.  

:warning:  The sign convention for the loads output in the yaml files is:
  - Fz positive spanwise
  - Mx positive towards suction side
  - My positive towards trailing edge


## Case 8 - Load extrapolation from 1D to 3D

- `transfer_loads` takes a reference 3D solution and a 1D loading and extrapolates it to 3D forces. Works for DEL, extreme or nominal loads. See bottom of that file for user definitions. 


## Case 9 - Various processing routines, mainly to handle high-fidelity outputs and cricle back to case 7

- `driver` is a minimal version of the driver from case 7, just to output the loads measured by OpenFAST in a certain wind inflow condition, and output "nominal" loadings.

- `structure_HiFi2LoFi`  converts the `.dat` output file of a high-fidelity run to a `.yaml` model file, basically adapting the thickness of all the structural zones based on  the results of the high-fidelity optimization. 

- `structure_Hst2HiFi`  converts the DVs of a `.hst` output file back into the `.dat` format that is needed to restart a high-fidelity simulation with the corresponding DVs.

- `plotLoFiFailure` plots the failure criterion evaluated by WISDEM. Optionally compare it with results from high-fidelity simulations.


Typically, you can use the driver as follows to compare high-fidelity and low-fidelity evaluations of the failure criteria:
- match your hifi and lofi structural models (in terms of geometry description), run the driver that outputs nominal loads, run case8 to extrapolate load, run a HiFi structural analysis with the extrapolated loads, use the structure_HiFi2LoFi to plot constraint in HiFi, use plotLoFiFailure to plot contraint in LoFi and compare.



## Procedure to apply do a WISDEM optimization with OpenFAST loads


Go to case 7, adapt the driver script and then run
```
python driver.py
```


To compare the result of 2 optimizations (or initial vs optimized), you can use
```
compare_designs --modeling_options modeling_options.yaml Madsen2019_10_forWEIS.yaml outputs_struct/blade_out.yaml
mv outputs outputs_withoutFatigue
```



## Procedure to apply the Combine Fidelity Approach

- Derive your original lofi model matching the IC hifi with correct thickness
- nominal test
    - Use driver of 09 to simulate nominal loads (see 06)
    - Transfer the loads to 3D using 08.
    - Simulate struct at nominal load; use struct_hifi2lofi to process the constraint; use plotLofiConstr to compare with lofi 
- Damage test
    - Use driver of 07 to simulate a 1y-eq DEL for a single vel 9m/s
    - Transfer the loads to 3D using 08.
    - Simulate struct the 1yr damage; use struct_hifi2lofi to process the constraint; use plotLofiConstr to compare with lofi
- Full optim
    - iter1
        - Simulate the full DLCs set with case 7
        - Transfer the loads to 3D using 08
        - Run the HiFi optim (don’t forget to copy/symlink the aero loads)
        - Feedback the structural changes into lofi: strucutre_hifi2lofi > updated_turbine.yaml
    - Iter2
        - Simulate the full DLCs set, with the updated yaml
        - Transfer the loads to 3D using 08
        - Run the HiFi optim with the DVGroup.dat file from the last iteration as initial condition
        - Feedback the structural changes into lofi: strucutre_hifi2lofi > updated_turbine.yaml
    - ...
    - Use 07/plot_results_HiFi to check how iter/DEL converge
