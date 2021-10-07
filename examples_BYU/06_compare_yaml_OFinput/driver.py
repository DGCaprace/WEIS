import os
import yaml

from wisdem import run_wisdem
from weis.aeroelasticse.runFAST_pywrapper   import runFAST_pywrapper, runFAST_pywrapper_batch
from weis.aeroelasticse.CaseGen_IEC         import CaseGen_IEC
from wisdem.commonse.mpi_tools              import MPI
from wisdem.inputs import load_yaml, write_yaml #, validate_without_defaults, validate_with_defaults, simple_types

from weis.glue_code.runWEIS import run_weis
from pCrunch.io import OpenFASTOutput

import sys, os, platform
import numpy as np
import matplotlib.pyplot as plt


# ---------------------
def my_write_yaml(instance, foutput):
    # Write yaml with updated values
    with open(foutput, "w", encoding="utf-8") as f:
        yaml.dump(instance, f)



#==================== DEFINITIONS  =====================================

## File management
mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
# fname_wt_input = mydir + os.sep + "IEA-10-198-RWT.yaml"
fname_wt_input = mydir + os.sep + "Madsen2019_10_forWEIS.yaml"
fname_modeling_options = mydir + os.sep + "modeling_options.yaml"
# fname_analysis_options = mydir + os.sep + "analysis_options_struct.yaml"
fname_analysis_options_WEIS = mydir + os.sep + "analysis_options_WEIS.yaml"


#location of servodyn lib (./local of weis)
run_dir1            = "/Users/dg/Documents/BYU/devel/Python/WEIS"

# run_dir2            = mydir + "/examples/01_aeroelasticse/" #os.path.dirname( os.path.realpath(__file__) ) + os.sep
run_dir2            = mydir + os.sep + ".." + os.sep + "Madsen2019_model_BD"



#==================== RUN THE TURBINE WITH WEIS, FROM YAML INPUTS =====================================
# Loading the wisdem/weis compatible yaml, and propagate information to aeroelasticse

# Could use WEIS and instructing the use of Level3=openFAST + iec
# OR
# Could code in this routine the call to IEC and AeroelasticSE manually.
# => We chose to use WEIS for maintenance purpose: we will define input files and then benefit from all WEIS functionalities.

# Run the base simulation
wt_opt, modeling_options, opt_options = run_weis(
    fname_wt_input, fname_modeling_options, fname_analysis_options_WEIS
)
# # In case one needs to overwrite data AFTER having read the yaml:
# # Construct a dict with values to overwrite
# overridden_values = {}
# overridden_values["rotorse.wt_class.V_mean_overwrite"] = 11.5
# # Run the modified simulation with the overwritten values
# wt_opt, modeling_options, opt_options = run_weis(
#     fname_wt_input,
#     fname_modeling_options,
#     fname_analysis_options_WEIS,
#     overridden_values=overridden_values,
# )

ref_V = wt_opt['rotorse.rp.powercurve.V']
ref_RPM = wt_opt['rotorse.rp.powercurve.Omega']
ref_pitch = np.array(wt_opt['rotorse.rp.powercurve.pitch']) #caution: negative values even though 

RPM_weis = wt_opt['aeroelastic.Omega_out']
Cp_weis  = wt_opt['aeroelastic.Cp_out']  #aero Cp
P_weis  = wt_opt['aeroelastic.P_out']
Pitch_weis = wt_opt['aeroelastic.pitch_out']


print("\n\n\n  -------------- DONE WITH WEIS ------------------\n")
print("\n\n\n")



#==================== RUN THE TURBINE WITH OPENFAST, USING ORIGINAL FAST INPUT FILES =====================================
# Unsteady loading computation from DLCs

# Turbine inputs
iec = CaseGen_IEC()
iec.Turbine_Class       = 'I'   # Wind class I, II, III, IV
iec.Turbulence_Class    = 'B'   # Turbulence class 'A', 'B', or 'C'
iec.D                   = 178.3  # Rotor diameter to size the wind grid
iec.z_hub               = 119.  # Hub height to size the wind grid
cut_in                  = 4.    # Cut in wind speed
cut_out                 = 25.   # Cut out wind speed
n_ws                    = 3    # Number of wind speed bins
TMax                    = 60.    # Length of wind grids and OpenFAST simulations, suggested 720 s
Vrated                  = 9.0 # Rated wind speed
Ttrans                  = max([0., TMax - 60.])  # Start of the transient for DLC with a transient, e.g. DLC 1.4
TStart                  = max([0., TMax - 600.]) # Start of the recording of the channels of OpenFAST


# Initial conditions to start the OpenFAST runs
u_ref     = np.array([5.0, 8.0, 10.0, 12.0, 15.]) # Wind speed 
pitch_ref = [0.0, 0.0, 0.0, 0.0, 0.0] # Pitch values in deg  
omega_ref = [6.0, 6.69, 8.36, 9.6, 9.6] # Rotor speeds in rpm
# u_ref     = ref_V
# pitch_ref = np.maximum(ref_pitch,np.zeros(len(ref_pitch))) #CAUTION: negative values!!
# omega_ref = ref_RPM


iec.init_cond = {}
iec.init_cond[("ElastoDyn","RotSpeed")]        = {'U':u_ref}
iec.init_cond[("ElastoDyn","RotSpeed")]['val'] = omega_ref
iec.init_cond[("ElastoDyn","BlPitch1")]        = {'U':u_ref}
iec.init_cond[("ElastoDyn","BlPitch1")]['val'] = pitch_ref
iec.init_cond[("ElastoDyn","BlPitch2")]        = iec.init_cond[("ElastoDyn","BlPitch1")]
iec.init_cond[("ElastoDyn","BlPitch3")]        = iec.init_cond[("ElastoDyn","BlPitch1")]


# DLC inputs

#   #bunch of them:
#   wind_speeds = np.linspace(int(cut_in), int(cut_out), int(n_ws))
#   iec.dlc_inputs = {}
#   iec.dlc_inputs['DLC']   = [1.1, 1.3, 1.4, 1.5, 5.1, 6.1, 6.3]
#   iec.dlc_inputs['U']     = [wind_speeds, wind_speeds,[Vrated - 2., Vrated, Vrated + 2.],wind_speeds, [Vrated - 2., Vrated, Vrated + 2., cut_out], [], []]
#   iec.dlc_inputs['Seeds'] = [[1],[1],[],[],[1],[1],[1]]
#   # iec.dlc_inputs['Seeds'] = [range(1,7), range(1,7),[],[], range(1,7), range(1,7), range(1,7)]
#   iec.dlc_inputs['Yaw']   = [[], [], [], [], [], [], []]
#   iec.PC_MaxRat           = 2.
#only power curve:
wind_speeds = [5.0, 8.0, 10.0, 12.0, 15.]
iec.dlc_inputs = {}
iec.dlc_inputs['DLC']   = [1.1]
iec.dlc_inputs['U']     = [wind_speeds]
iec.dlc_inputs['Seeds'] = [[1]]
# iec.dlc_inputs['Seeds'] = [range(1,7), range(1,7),[],[], range(1,7), range(1,7), range(1,7)]
iec.dlc_inputs['Yaw']   = [[]]  


iec.TStart              = Ttrans
iec.TMax                = TMax    # wind file length
iec.transient_dir_change        = 'both'  # '+','-','both': sign for transient events in EDC, EWS
iec.transient_shear_orientation = 'both'  # 'v','h','both': vertical or horizontal shear for EWS

# Management of parallelization
if MPI:
    from wisdem.commonse.mpi_tools import map_comm_heirarchical, subprocessor_loop, subprocessor_stop
    n_OF_runs = 0
    for i in range(len(iec.dlc_inputs['DLC'])):
        # Number of wind speeds
        if iec.dlc_inputs['DLC'][i] == 1.4: # assuming 1.4 is run at [V_rated-2, V_rated, V_rated] and +/- direction change
            if iec.dlc_inputs['U'][i] == []:
                n_U = 6
            else:
                n_U = len(iec.dlc_inputs['U'][i]) * 2
        elif iec.dlc_inputs['DLC'][i] == 5.1: # assuming 5.1 is run at [V_rated-2, V_rated, V_rated]
            if iec.dlc_inputs['U'][i] == []:
                n_U = 3
            else:
                n_U = len(iec.dlc_inputs['U'][i])
        elif iec.dlc_inputs['DLC'][i] in [6.1, 6.3]: # assuming V_50 for [-8, 8] deg yaw error
            if iec.dlc_inputs['U'][i] == []:
                n_U = 2
            else:
                n_U = len(iec.dlc_inputs['U'][i])
        else:
            n_U = len(iec.dlc_inputs['U'][i])
        # Number of seeds
        if iec.dlc_inputs['DLC'][i] == 1.4: # not turbulent
            n_Seeds = 1
        else:
            n_Seeds = len(iec.dlc_inputs['Seeds'][i])
        n_OF_runs += n_U*n_Seeds
        available_cores = MPI.COMM_WORLD.Get_size()
        n_parallel_OFruns = np.min([available_cores - 1, n_OF_runs])
        comm_map_down, comm_map_up, color_map = map_comm_heirarchical(1, n_parallel_OFruns)
        sys.stdout.flush()

# Naming, file management, etc
iec.wind_dir        = 'outputs_FAST/wind'
iec.case_name_base  = 'Madsen10'
if MPI:
    iec.cores = available_cores
else:
    iec.cores = 1

iec.debug_level = 2
if MPI:
    iec.parallel_windfile_gen = True
    iec.mpi_run               = True
    iec.comm_map_down         = comm_map_down
else:
    iec.parallel_windfile_gen = False
    iec.mpi_run               = False
iec.run_dir = 'outputs_FAST/Madsen10'

# Run case generator / wind file writing
case_inputs = {}
case_inputs[("Fst","TMax")]              = {'vals':[TMax], 'group':0}
case_inputs[("Fst","TStart")]            = {'vals':[TStart], 'group':0}
case_inputs[("Fst","DT")]                = {'vals':[0.01], 'group':0}
case_inputs[("Fst","DT_Out")]            = {'vals':[0.01], 'group':0}  #0.005  
# case_inputs[("Fst","OutFileFmt")]        = {'vals':[2], 'group':0}   #<------------- binary output
case_inputs[("Fst","OutFileFmt")]        = {'vals':[1], 'group':0}  #<------------- ASCII output for debug
# case_inputs[("Fst","CompHydro")]         = {'vals':[1], 'group':0}
# case_inputs[("Fst","CompSub")]           = {'vals':[0], 'group':0}
case_inputs[("InflowWind","WindType")]   = {'vals':[1], 'group':0}
case_inputs[("ElastoDyn","TwFADOF1")]    = {'vals':["False"], 'group':0}
case_inputs[("ElastoDyn","TwFADOF2")]    = {'vals':["False"], 'group':0}
case_inputs[("ElastoDyn","TwSSDOF1")]    = {'vals':["False"], 'group':0}
case_inputs[("ElastoDyn","TwSSDOF2")]    = {'vals':["False"], 'group':0}
case_inputs[("ElastoDyn","FlapDOF1")]    = {'vals':["True"], 'group':0}
case_inputs[("ElastoDyn","FlapDOF2")]    = {'vals':["True"], 'group':0}
case_inputs[("ElastoDyn","EdgeDOF")]     = {'vals':["True"], 'group':0}
case_inputs[("ElastoDyn","DrTrDOF")]     = {'vals':["False"], 'group':0}
# case_inputs[("ElastoDyn","GenDOF")]      = {'vals':["False"], 'group':0}
case_inputs[("ElastoDyn","YawDOF")]      = {'vals':["False"], 'group':0}
case_inputs[("ElastoDyn","PtfmSgDOF")]   = {'vals':["False"], 'group':0}
case_inputs[("ElastoDyn","PtfmSwDOF")]   = {'vals':["False"], 'group':0}
case_inputs[("ElastoDyn","PtfmHvDOF")]   = {'vals':["False"], 'group':0}
case_inputs[("ElastoDyn","PtfmRDOF")]    = {'vals':["False"], 'group':0}
case_inputs[("ElastoDyn","PtfmPDOF")]    = {'vals':["False"], 'group':0}
case_inputs[("ElastoDyn","PtfmYDOF")]    = {'vals':["False"], 'group':0}
# case_inputs[("ServoDyn","PCMode")]       = {'vals':[5], 'group':0}
# case_inputs[("ServoDyn","VSContrl")]     = {'vals':[5], 'group':0}

# if platform.system() == 'Windows':
#     path2dll = os.path.join(run_dir1, 'local/lib/libdiscon.dll')
# elif platform.system() == 'Darwin':
#     path2dll = os.path.join(run_dir1, 'local/lib/libdiscon.dylib')
# else:
#     path2dll = os.path.join(run_dir1, 'local/lib/libdiscon.so')

# case_inputs[("ServoDyn","DLL_FileName")] = {'vals':[path2dll], 'group':0}
# case_inputs[("AeroDyn15","AFAeroMod")]   = {'vals':[2], 'group':0}  #--> turn on Leishman-Beddoes
# case_inputs[("AeroDyn15","TwrAero")]     = {'vals':["True"], 'group':0}
# case_inputs[("AeroDyn15","TwrPotent")]   = {'vals':[1], 'group':0}
# case_inputs[("AeroDyn15","TwrShadow")]   = {'vals':[1], 'group':0}

channels = {}
for var in ["TipDxc1", "TipDyc1", "TipDzc1", "TipDxb1", "TipDyb1", "TipDxc2", "TipDyc2", "TipDzc2", "TipDxb2", "TipDyb2", "TipDxc3", "TipDyc3", "TipDzc3", "TipDxb3", "TipDyb3", "RootMxc1", "RootMyc1", "RootMzc1", "RootMxb1", "RootMyb1", "RootMxc2", "RootMyc2", "RootMzc2", "RootMxb2", "RootMyb2", "RootMxc3", "RootMyc3", "RootMzc3", "RootMxb3", "RootMyb3", "TwrBsMxt", "TwrBsMyt", "TwrBsMzt", "GenPwr", "GenTq", "RotThrust", "RtAeroCp", "RtAeroCt", "RotSpeed", "BldPitch1", "TTDspSS", "TTDspFA", "NacYaw", "Wind1VelX", "Wind1VelY", "Wind1VelZ", "LSSTipMxa","LSSTipMya","LSSTipMza","LSSTipMxs","LSSTipMys","LSSTipMzs","LSShftFys","LSShftFzs", "TipRDxr", "TipRDyr", "TipRDzr"]:
    channels[var] = True

#TODO
#DG: hardcoded node outputs in FASTwriter
#DG: hardcoded DELs based on those outputs in runFAST_pywrapper

# Parallel file generation with MPI
if MPI:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
else:
    rank = 0
if rank == 0:
    case_list, case_name_list, dlc_list = iec.execute(case_inputs=case_inputs)

    print(case_name_list)
    # print(case_list)

    #for var in var_out+[var_x]:

    for case in case_list: #FORCE GENDOF TO FALSE, OTHERWISE IEC PUT IT BACK ON ABOVE
        case[("ElastoDyn","GenDOF")]   = False
        # case[("ElastoDyn","FlapDOF1")] = False
        # case[("ElastoDyn","FlapDOF2")] = False
        # case[("ElastoDyn","EdgeDOF")]  = False
        print(case)

    # Run FAST cases
    fastBatch                   = runFAST_pywrapper_batch(FAST_ver='OpenFAST',dev_branch = True)

#    #Fixed-bottom
    fastBatch.FAST_InputFile    = 'DTU_10MW.fst'   # FAST input file (ext=.fst)
    fastBatch.FAST_directory    = run_dir2   # Path to fst directory files

    fastBatch.channels          = channels
    fastBatch.FAST_runDirectory = iec.run_dir
    fastBatch.case_list         = case_list
    fastBatch.case_name_list    = case_name_list
    fastBatch.debug_level       = 2

    fastBatch.keep_time         = True  #DG


    if MPI:
        summary_stats, extreme_table, DELs, ct = fastBatch.run_mpi(comm_map_down)
    else:
        summary_stats, extreme_table, DELs, ct = fastBatch.run_serial()

if MPI:
    sys.stdout.flush()
    if rank in comm_map_up.keys():
        subprocessor_loop(comm_map_up)
    sys.stdout.flush()

# Close signal to subprocessors
if rank == 0 and MPI:
    subprocessor_stop(comm_map_down)

print("\n\n\n  -------------- DONE WITH FAST ------------------\n\n\n\n")



#==================== DO SOME POSTPROCESSING =====================================

print(" -------------- WEIS SUMMARY: ------------------\n")

# print(f"ref V    : {ref_V     }")
# print(f"ref RPM  : {ref_RPM   }")
# print(f"ref pitch: {ref_pitch }")

print(f"avg RPM  : {RPM_weis}")
print(f"avg Cp   : {Cp_weis}")
print(f"avg P    : {P_weis}")
print(f"avg pitch: {Pitch_weis}")
print(f"AEP      : {wt_opt['aeroelastic.AEP']}")
print(f"tip defl : {wt_opt['aeroelastic.summary_stats']['TipDxc1']['mean']}")

print(" -------------- FAST SUMMARY: ------------------\n")

# GenPwr unavailable since I did not use servodyn.

Cp_myfast = summary_stats["RtAeroCp"]["mean"]
RPM_myfast = summary_stats["RotSpeed"]["mean"]
Pitch_myfast = summary_stats["BldPitch1"]["mean"]
TipDefl_myfast = summary_stats["TipDxc1"]["mean"]

print(f"avg RPM  : {RPM_myfast}")
print(f"avg Cp   : {Cp_myfast}")
print(f"avg pitch: {Pitch_myfast}")
print(f"tip defl : {TipDefl_myfast}")

#  "GenTq", "RotThrust", "RtAeroCp"


## Compare aero Cp ##

# my openfast model predicts a bit more Cp...
# Factors that do slightly contribute:
#   - tried to use same skewness model : did not improve
#   - as is, the WEIS structural model is more rigid so leads to 5m tip deflection instead of 7 in my openfast model
#   - the rotor radius is not exactly the same, with a difference of about 0.1m?
# Factors that DO contribute:
#   - the disagreement at 15 m/s is due to the fact that ROSCO in WEIS started feather the blade to maintain 10MW, whereas my fast stays at pitch = 0
#   - the controller in WEIS recurrently leads to the turbine rotating a bit slower, which readily translates into a smaller Cp 
#   - the OpenFAST simulation uses turbulent DLC 1.1 so the avg upstream velocity may not be exactly what it should (this explain the increased discrepancy at low vel: 5m/s)
#   - using WEIS generated polars:
#       - with WEIS aero blade file, I have +~0.01 in Cp. May be due to differences in the blade file OR since I use the same indices for airfoils, simply due to the fact that the r locations are not the same.
#       - most of the difference comes from the polars!!  WEIS do a smart interpolation leading to a unique polar at every station, whereas original model has sharp transitions !
#
# Illustration: see the ./plot subdir:
#    I replaced some portions of my openFAST model with files produced with WEIS. 
#    All figs in this folder are done with the same ED file, all DOF being FALSE. Sims are short (10 sec)
#    - weisPolars_myBlade: modified my AD file to use the 40 polars from WEIS, still using my blade file. Porblem doing that: r locations in my blade file and in weis' are not the same
#    - weisPolars_weisBlade: the closest I can make my model from WEIS The remaining discrepancy are due to slightly off upstream velocity in OpenFAST due to turbulence, and rotational velocity in WEIS due to controller.
#    - 1Polars_myBlade: just to make sure there is nothing weird with polar interpolation, use only the FFW-241 for the whole blade. My blade file.
#    - 1Polars_weisBlade: also only 1 polar. We can still see a ~0.01 difference in Cp, which can only come from the blade file which is reinterpolated uniformly in WEIS.
# Conclusion: what makes the most difference is the interpolated polar, and the reinterpolated blade file.

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

ax.plot(wind_speeds,Cp_weis,'x-', label='WEIS')
ax.plot(wind_speeds,Cp_myfast,'o-', label='my OpenFAST') 
plt.xlabel("U [m/s]")
plt.ylabel("Cp")
plt.legend()
plt.show()