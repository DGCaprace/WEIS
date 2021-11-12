import os
import yaml

from wisdem import run_wisdem
from wisdem.inputs import load_yaml, write_yaml #, validate_without_defaults, validate_with_defaults, simple_types
# from pCrunch import PowerProduction, LoadsAnalysis
# from pCrunch.io import OpenFASTAscii, OpenFASTBinary#, OpenFASTOutput

import sys, shutil
import numpy as np
import matplotlib.pyplot as plt


"""This script starts from a given turbine yaml and an external loading file, and just computes 1 evaluation in WISDEM.

:raises FileNotFoundError: [description]
:return: [description]
:rtype: [type]
"""

# ---------------------
def my_write_yaml(instance, foutput):
    if os.path.isfile(foutput):
        print(f"File {foutput} already exists... replacing it.")
        os.remove(foutput)
    # Write yaml with updated values
    with open(foutput, "w", encoding="utf-8") as f:
        yaml.dump(instance, f)


#==================== DEFAULTS  =====================================
withEXTR = False
withDEL = False  
withNominal = False #REPLACE the EXTR with the nominal load
WRONG_CONVENTION = False

#==================== DEFINITIONS  =====================================

## File management
mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
fname_modeling_options = mydir + os.sep + "modeling_options.yaml"
fname_analysis_options = mydir + os.sep + "analysis_options_struct.yaml"

# fname_wt_input = mydir + os.sep + "Madsen2019_10_forWEIS_isotropic.yaml"
# fname_loads = mydir + os.sep + "Madsen2019_10_forWEIS_isotropic_ED/nominalLoads.yaml" #the loading condition you want to use (should have a nominal section)
# folder_arch = mydir + os.sep + "LoFiEval_isotropic_nominalLoads"
# withNominal = True

# fname_wt_input = mydir + os.sep + "Madsen2019_10_forWEIS.yaml"
# fname_loads = mydir + os.sep + "Madsen2019_10_forWEIS/nominalLoads.yaml" #the loading condition you want to use (should have a nominal section)
# folder_arch = mydir + os.sep + "LoFiEval_composite_nominalLoads"
# withNominal = True

fname_wt_input = mydir + os.sep + "Madsen2019_10_forWEIS_isotropic.yaml"
fname_loads = mydir + os.sep + "../07_test_iterateDEL/results-IEC1.1-IEC1.3_5vels_120s_0Glob_norm_neq1/analysis_options_struct_withDEL.yaml" 
folder_arch = mydir + os.sep + "LoFiEval_isotropic_DEL"
withDEL = True
WRONG_CONVENTION = True
m_wohler = 10
n_life_eq = 1


runWISDEM = True


#==================== ======== =====================================
## Preprocessing: filling in the loads and relevant parameters

if not os.path.isdir(folder_arch):
    runWISDEM = True

if withEXTR and withDEL and withNominal:
    raise ValueError("choose one of EXTRM, DEL or norminal")

if withEXTR and withNominal:
    raise ValueError("can't do withEXTR and withNominal: nominal loads actually replace the extreme loads")

# analysis_opt = load_yaml(fname_analysis_options)
# wt_init = load_yaml(fname_wt_input)
# modeling_options = load_yaml(fname_modeling_options)  #initial load


## Load the loading file
schema_loads = load_yaml(fname_loads)

## Update the analysis file
schema = load_yaml(fname_analysis_options)


if withEXTR or withNominal:
    if withEXTR:
        src_name = "extreme"
    else: 
        src_name = "nominal"

    schema["extreme"] = {}
    schema["extreme"]["description"] = schema_loads[src_name]["description"]
    schema["extreme"]["grid_nd"] = schema_loads[src_name]["grid_nd"]
    if WRONG_CONVENTION:
        #sign change was done but should not have swapped
        schema["extreme"]["deMLx"] = ( np.array(schema_loads[src_name]["deMLy"]) ).tolist()
        schema["extreme"]["deMLy"] = ( np.array(schema_loads[src_name]["deMLx"]) ).tolist()
    else:
        schema["extreme"]["deMLx"] = schema_loads[src_name]["deMLx"]
        schema["extreme"]["deMLy"] = schema_loads[src_name]["deMLy"]
    schema["extreme"]["deFLz"] = schema_loads[src_name]["deFLz"]
    schema["constraints"]["blade"]["extreme_loads_from_user_inputs"] = True #we are using that channel as a way to specify a loading. The we will read the corresponding strain the EXTRM strain  output




if withDEL:
    schema["DEL"] = {}
    schema["DEL"]["description"] = schema_loads["DEL"]["description"]
    schema["DEL"]["grid_nd"] = schema_loads["DEL"]["grid_nd"]
    if WRONG_CONVENTION:
        #sign change was done but should not have swapped
        schema["DEL"]["deMLx"] = ( np.array(schema_loads["DEL"]["deMLy"]) ).tolist()
        schema["DEL"]["deMLy"] = ( np.array(schema_loads["DEL"]["deMLx"]) ).tolist()
    else:
        schema["DEL"]["deMLx"] = schema_loads["DEL"]["deMLx"]
        schema["DEL"]["deMLy"] = schema_loads["DEL"]["deMLy"]
    
    schema["DEL"]["deFLz"] = schema_loads["DEL"]["deFLz"]

    schema["constraints"]["blade"]["fatigue_spar_cap_ss"]["flag"] = True
    schema["constraints"]["blade"]["fatigue_spar_cap_ps"]["flag"] = True
    schema["constraints"]["blade"]["fatigue_spar_cap_ss"]["eq_Ncycle"] = float(n_life_eq)
    schema["constraints"]["blade"]["fatigue_spar_cap_ps"]["eq_Ncycle"] = float(n_life_eq)
    schema["constraints"]["blade"]["fatigue_spar_cap_ss"]["m_wohler"] = m_wohler
    schema["constraints"]["blade"]["fatigue_spar_cap_ps"]["m_wohler"] = m_wohler

schema["general"]["folder_output"] = folder_arch

fname_analysis_options_struct = mydir + os.sep + "analysis_options_struct_withMyLoading.yaml"
my_write_yaml(schema, fname_analysis_options_struct)


#==================== ======== =====================================
# Simulation

if runWISDEM:
    wt_opt, analysis_options, opt_options = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options_struct,
        # overridden_values = overridden_values
    )

    print("\n\n\n  -------------- DONE WITH WISDEM ------------------\n\n\n\n")    

#==================== ======== =====================================
# Read the outputs

max_strain = schema["constraints"]["blade"]["strains_spar_cap_ss"]["max"]

WISDEMout = folder_arch + "/blade_out.npz"

fig3, ax3 = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
fig4, ax4 = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

with np.load(WISDEMout) as a:

    r = np.array(a["rotorse.rs.z_az_m"])
    r = (r-r[0])/(r[-1]-r[0])
    
    if withNominal or withEXTR:
        #My passed loading stuff:
        data3 = a["rotorse.rs.extreme_strains.F3_N"]  #should be = to my input
        data4 = a["rotorse.rs.extreme_strains.M1_N*m"]  #should be = to my input (well, my input but set in principal axes)
        data5 = a["rotorse.rs.extreme_strains.M2_N*m"]  #should be = to my input (well, my input but set in principal axes)
        data1 = a["rotorse.rs.extreme_strains.strainU_spar"] / max_strain #rebuild the failure constraint
        data2 = a["rotorse.rs.extreme_strains.strainL_spar"] / max_strain #rebuild the failure constraint

    if withDEL:
        #My passed loading stuff:
        data3 = a["rotorse.rs.fatigue_strains.F3_N"]  #should be = to my input
        data4 = a["rotorse.rs.fatigue_strains.M1_N*m"]  #should be = to my input (well, my input but set in principal axes)
        data5 = a["rotorse.rs.fatigue_strains.M2_N*m"]  #should be = to my input (well, my input but set in principal axes)
        data1 = a["rotorse.rs.fatigue_strains.strainU_spar"] / max_strain #rebuild the failure constraint
        data2 = a["rotorse.rs.fatigue_strains.strainL_spar"] / max_strain #rebuild the failure constraint
    
    # #original gust stuff:
    # data = a["rotorse.rs.strains.F3_N"]
    # data = a["rotorse.rs.strains.M1_N*m"]
    # data = a["rotorse.rs.strains.M2_N*m"]
    # data1 = a["rotorse.rs.strains.strainU_spar"] / max_strain
    # data2 = a["rotorse.rs.strains.strainL_spar"] / max_strain

    # r = range(4)
    # data = a["rotorse.rs.constr.constr_max_strainU_spar"]
    # data = a["rotorse.rs.constr.constr_max_strainL_spar"]

    ax3.plot(r,-data1,'o-', label=f'SS')  
    ax3.plot(r, data2,'x-', label=f'PS') #NEGATIVE to make them both positive??? but the strain should be > 0 on the PS!!??

    ax4.plot(r,data3,'-', label=f'F3')
    ax4.plot(r,data4,'-', label=f'M1')
    ax4.plot(r,data5,'-', label=f'M2')

ax3.legend()
ax3.set_xlabel("r/R")
ax4.legend()
ax4.set_xlabel("r/R")

plt.show()