import os
import yaml

from wisdem.inputs import load_yaml #, write_yaml #, validate_without_defaults, validate_with_defaults, simple_types

import sys, shutil
import numpy as np
import matplotlib.pyplot as plt


#==================== DEFINITIONS  =====================================

## File management
mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
fname_wt_input = mydir + os.sep + "Madsen2019_10_forWEIS.yaml"
fname_modeling_options = mydir + os.sep + "modeling_options.yaml"
fname_analysis_options = mydir + os.sep + "analysis_options_struct.yaml"
fname_analysis_options_WEIS = mydir + os.sep + "analysis_options_WEIS.yaml"

folder_arch = mydir + os.sep + "results-IEC1.1_5vels_120s_4Glob"
folder_arch = mydir + os.sep + "results-60sec_TMP"
# folder_arch = mydir + os.sep + "results-120sec_TMP"
folder_arch = mydir + os.sep + "results__1vel_600s"  #-> before fixing wrong structural twist 
folder_arch = mydir + os.sep + "results__1vel_120s_fix" #-> after fixing wrong structural twist 
folder_arch = mydir + os.sep + "results"

nGlobalIter = 4

plotRelativeDEL = False

sparCapSS_name = "DP13_DP10_uniax"
sparCapPS_name = "DP07_DP04_uniax"

# --- prepare plots ---

#DEL distro:
fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
plt.xlabel("r/R")
plt.ylabel("thickness [m]")

fig2, ax2s = plt.subplots(nrows=3, ncols=1, figsize=(10, 5))
plt.xlabel("r/R")
if plotRelativeDEL:
    ax2s[0].set_ylabel("DEM1_i / DEM1_0")
    ax2s[1].set_ylabel("DEM2_i / DEM2_0")
    ax2s[2].set_ylabel("DEF3_i / DEF3_0")
else:
    ax2s[0].set_ylabel("DEM1 [N]")
    ax2s[1].set_ylabel("DEM2 [N]")
    ax2s[2].set_ylabel("DEF3 [N/m]")

#==================== LOAD DATA AND PLOT =====================================

for IGLOB in range(nGlobalIter): 
    #load
    curr_iter = f"iter_{IGLOB}"
    blade_out = load_yaml(folder_arch + os.sep + "outputs_optim" + os.sep + curr_iter + os.sep + "blade_out.yaml")
    analysis  = load_yaml(folder_arch + os.sep + "outputs_optim" + os.sep + curr_iter + os.sep + "blade_out-analysis.yaml")

    roR_tSS = []
    thickSS = []
    roR_tPS = []
    thickPS = []
    for layer in blade_out["components"]["blade"]["internal_structure_2d_fem"]["layers"]:
        if sparCapSS_name in layer["name"]:
            roR_tSS = layer["thickness"]["grid"]
            thickSS = layer["thickness"]["values"]
        if sparCapPS_name in layer["name"]:
            roR_tPS = layer["thickness"]["grid"]
            thickPS = layer["thickness"]["values"]

    if not thickSS:
        print("Could not find Spar cap suction side")
    if not thickPS:
        print("Could not find Spar cap pressure side")

    roR_d = analysis["DEL"]["grid_nd"]
    deML1 = np.array(analysis["DEL"]["deMLx"])
    deML2 = np.array(analysis["DEL"]["deMLy"])
    deFL3 = np.array(analysis["DEL"]["deFLz"])
    if IGLOB ==0:
        deML1_0 = deML1
        deML2_0 = deML2
        deFL3_0 = deFL3


    #plot
    ss = ax1.plot(roR_tSS,thickSS,'x-', label=f'SparCapSS i{IGLOB}')
    ps = ax1.plot(roR_tPS,thickPS,'o--', label=f'SparCapPS i{IGLOB}', color=ss[0].get_color())

    if plotRelativeDEL:
        ax2s[0].plot(roR_d,deML1/deML1_0,'x-', label=f'i{IGLOB}') 
        ax2s[1].plot(roR_d,deML2/deML2_0,'x-', label=f'i{IGLOB}') 
        ax2s[2].plot(roR_d,deFL3/deFL3_0,'x-', label=f'i{IGLOB}') 
    else:
        ax2s[0].plot(roR_d,deML1,'x-', label=f'i{IGLOB}') 
        ax2s[1].plot(roR_d,deML2,'x-', label=f'i{IGLOB}') 
        ax2s[2].plot(roR_d,deFL3,'x-', label=f'i{IGLOB}') 

ax1.legend()
ax2s[2].legend()

# plt.tight_layout()
fig1.savefig(folder_arch + "/thickness.png")
fig2.savefig(folder_arch + "/DELs.png")



#==================== LOAD DATA AND PLOT FROM OPTIMIZATION =====================================

WISDEMout = folder_arch + "/outputs_optim/iter_%i/blade_out.npz"

fig3, ax3 = plt.subplots(nrows=2, ncols=1, figsize=(10, 5))
plt.xlabel("r/R")

for IGLOB in range(nGlobalIter): 
    with np.load(WISDEMout%(IGLOB)) as a:

        r = np.array(a["rotorse.rs.z_az_m"])
        r = (r-r[0])/(r[-1]-r[0])
        
        #DEL stuff:
        # data = a["rotorse.rs.fatigue_strains.M1_N*m"]
        # data = a["rotorse.rs.fatigue_strains.M2_N*m"]
        # data = a["rotorse.rs.fatigue_strains.F3_N"]
        # data = a["rotorse.rs.fatigue_strains.strainU_spar"]
        # data = a["rotorse.rs.fatigue_strains.strainL_spar"]
        data1 = a["rotorse.rs.fatigue_strains.strainU_spar"] / 3500.e-6 #surrogate to damage constraint
        data2 = a["rotorse.rs.fatigue_strains.strainL_spar"] / 3500.e-6 #surrogate to damage constraint
        
        # #EXTRM stuff:
        # data1 = a["rotorse.rs.strains.M1_N*m"]
        # data2 = a["rotorse.rs.strains.M2_N*m"]
        # data = a["rotorse.rs.strains.F3_N"]
        # data = a["rotorse.rs.strains.strainU_spar"]
        # data = a["rotorse.rs.strains.strainL_spar"]
        # data = a["rotorse.rs.constr.constr_max_strainU_spar"]
        # data = a["rotorse.rs.constr.constr_max_strainL_spar"]
        # r = range(4)

        ax3[0].plot(r,data1,'x-', label=f'i{IGLOB}')
        ax3[1].plot(r,data2,'x-', label=f'i{IGLOB}')

ax3[0].legend()
plt.xlabel("r/R")

plt.show()
