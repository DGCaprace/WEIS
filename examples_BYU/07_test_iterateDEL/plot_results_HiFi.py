import os
import yaml
from yaml import Dumper, Loader
from pyoptsparse.pyOpt_history import History

import sys, shutil
import numpy as np
import matplotlib.pyplot as plt


# ==

def load_yaml(filepath):

    with open(filepath) as f:
        data = yaml.load(f, Loader=Loader)

    return data



#==================== DEFINITIONS  =====================================

## File management

# mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file

# folder_arch = mydir 
# DEL_folders = [ "results-IEC1.1_5vels_120s_0Glob_neq1",
#                 "results-IEC1.1_5vels_120s_1GlobHF_neq1",
#                 "results-IEC1.1_5vels_120s_2GlobHF_neq1",
#                 "results-IEC1.1_5vels_120s_3GlobHF_neq1",
# ]
# FIX_FOR_SIGN = -1

# folder_hst = "/Users/dg/Documents/BYU/simulation_data/ATLANTIS/MDAO/Aerostructural/Optimization/"
# HST_files = ["1pt_fatigue_ITER1_44949859_L3/Opt_output/SLSQP_hist_L31pt_fatigue_44949859.hst",
#             "1pt_fatigue_ITER2_45056517_L3/Opt_output/SLSQP_hist_L31pt_fatigue_45056517.hst",
#             "1pt_fatigue_ITER3_45066025_L3/Opt_output/SLSQP_hist_L31pt_fatigue_45066025.hst",
#             "1pt_fatigue_ITER4_45067084_L3/Opt_output/SLSQP_hist_L31pt_fatigue_45067084.hst",
# ]

# loads = ["DEL"]

##--------------

mydir = mydir = "/Users/dg/OneDrive - BYU/BYU_ATLANTIS/papers/2022_BestFidelity_BYU_UM/results/7_DLCs/"

folder_arch = mydir 
DEL_folders = [ "results-IEC1.1-IEC1.3-12vels-6s-IC",
                "results-IEC1.1-IEC1.3-12vels-6s-ITER1",
                "results-IEC1.1-IEC1.3-12vels-6s-ITER2",
                "results-IEC1.1-IEC1.3-12vels-6s-ITER3",
                "results-IEC1.1-IEC1.3-12vels-6s-ITER4",
]
FIX_FOR_SIGN = 1

folder_hst = "/Users/dg/OneDrive - BYU/BYU_ATLANTIS/papers/2022_BestFidelity_BYU_UM/results/MDAO/Aerostructural/"
HST_files = ["2pt_extreme_fatigue_45285270_L3/Opt_output/SNOPT_hist_L32pt_extreme_fatigue_45285270.hst",
             "2pt_extreme_fatigue_45302283_L3/Opt_output/SNOPT_hist_L32pt_extreme_fatigue_45302283.hst",
             "2pt_extreme_fatigue_45305271_L3/Opt_output/SNOPT_hist_L32pt_extreme_fatigue_45305271.hst",
             "2pt_extreme_fatigue_45308085_L3/Opt_output/SNOPT_hist_L32pt_extreme_fatigue_45308085.hst",
]

loads = ["DEL"]
# loads = ["extreme"]

#--

plotRelativeDEL = True
HST_fullHistory = True



# sparCapSS_name = "DP13_DP10_uniax"
# sparCapPS_name = "DP07_DP04_uniax"

# ===================================================================================
nGlobalIterDEL = len(DEL_folders)

# --- prepare plots ---
fig1, ax1 = plt.subplots(nrows=2, ncols=1, figsize=(10, 5))
plt.xlabel("r/R")
if plotRelativeDEL:
    ax1[0].set_ylabel(r"$DEF_n (i) \: / \: DEF_n (1)$")
    ax1[1].set_ylabel(r"$DEF_t (i) \: / \: DEF_t (1)$")
else:
    ax1[0].set_ylabel(r"$DEF_n \, [N/m]$")
    ax1[1].set_ylabel(r"$DEF_t \, [N/m]$")

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

for load in loads:
    for IGLOB in range(nGlobalIterDEL): 
        IGs1 = IGLOB+1
        #load
        curr_iter = DEL_folders[IGLOB]
        aero_loads = load_yaml(folder_arch + os.sep + curr_iter + os.sep + "aggregatedEqLoads.yaml")
        beam_loads = load_yaml(folder_arch + os.sep + curr_iter + os.sep + "analysis_options_struct_withDEL.yaml")

        roR_d = beam_loads[load]["grid_nd"]
        deML1 = np.array(beam_loads[load]["deMLx"])
        deML2 = np.array(beam_loads[load]["deMLy"])
        deFL3 = np.array(beam_loads[load]["deFLz"])
        deFn = np.array(aero_loads[load]["Fn"])
        deFt = np.array(aero_loads[load]["Ft"])

        if IGLOB ==0:
            deML1 *= FIX_FOR_SIGN
            deML2 *= FIX_FOR_SIGN
            deML1_0 = deML1
            deML2_0 = deML2
            deFL3_0 = deFL3
            deFn_0 = deFn
            deFt_0 = deFt

        if plotRelativeDEL:
            ax2s[0].plot(roR_d,deML1/deML1_0,'x-', label=f'i{IGs1}') 
            ax2s[1].plot(roR_d,deML2/deML2_0,'x-', label=f'i{IGs1}') 
            ax2s[2].plot(roR_d,deFL3/deFL3_0,'x-', label=f'i{IGs1}') 

            ax1[0].plot(roR_d,deFn/deFn_0,'x-', label=rf'$i={IGs1}$') 
            ax1[1].plot(roR_d,deFt/deFt_0,'x-', label=rf'$i={IGs1}$') 
        else:
            ax2s[0].plot(roR_d,deML1,'x-', label=f'i{IGs1}') 
            ax2s[1].plot(roR_d,deML2,'x-', label=f'i{IGs1}') 
            ax2s[2].plot(roR_d,deFL3,'x-', label=f'i{IGs1}') 

            ax1[0].plot(roR_d,deFn,'x-', label=rf'$i={IGs1}$') 
            ax1[1].plot(roR_d,deFt,'x-', label=rf'$i={IGs1}$') 

ax1[1].legend()
ax2s[2].legend()

# plt.tight_layout()

suff = ''
if plotRelativeDEL:
    suff = '_rel'
fig1.savefig(mydir +os.sep+ DEL_folders[-1] + f"/aero_loads{suff}.eps")
fig2.savefig(mydir +os.sep+ DEL_folders[-1] + f"/beam_loads{suff}.eps")



#==================== LOAD HHST DATA =====================================
nGlobalIter= len(HST_files)

if nGlobalIter==0:
    plt.show()
    exit()

if HST_fullHistory:
    fig3, ax3 = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
else:
    fig3, ax3 = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

ref_obj = 1.
lastNIter = 0

f_cnt = []
obj_val = []

for IGLOB in range(nGlobalIter): 

    #load
    hst = folder_hst+ os.sep + HST_files[IGLOB]
    print(hst)
    optHist = History(hst)
    histValues = optHist.getValues()

    # print(histValues.keys())
    # DVs_hst = histValues["struct"]
    obj_hst = histValues["obj"]

    if IGLOB == 0:
        ref_obj = obj_hst[0]
        obj_val.append(obj_hst[0])
    
    if HST_fullHistory:
        rng = range(lastNIter, lastNIter+ len(obj_hst))
        ax3.plot(rng,obj_hst / ref_obj,'-', label=f'i{IGLOB+1}')
        

    f_cnt.append(len(obj_hst))
    obj_val.append(obj_hst[-1])
    lastNIter += len(obj_hst)

print(obj_val)

if HST_fullHistory:
    ax3.legend()
    plt.xlabel("func. call")
    plt.ylabel("objective")    

    fig3.savefig(mydir +os.sep+ DEL_folders[-1] + f"/obj_hist.eps")
else:
    rng = range(0,nGlobalIter+1)
    ax3[0].plot(rng,np.array(obj_val) / ref_obj,'-x')
    ax3[0].set_xlabel("outer iter")
    ax3[0].set_ylabel("objective")    
    rng = range(1,nGlobalIter+1)
    ax3[1].plot(rng,f_cnt,'-x')
    ax3[1].set_ylabel("func. calls")    
    ax3[1].set_xlabel("outer iter")

    fig3.savefig(mydir +os.sep+ DEL_folders[-1] + f"/obj_hist_2.eps")

plt.show()
