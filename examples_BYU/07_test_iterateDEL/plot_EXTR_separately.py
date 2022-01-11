import os

# import sys, shutil
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from wisdem.inputs import load_yaml

""" 
# ==================== README ==========================

Plot of the loads experienced in various DLC simulations.
For extrapolated DLCs (1.1,1.3), the extrapolated loads are shown along with the average measured during the time serie, and the 1-sigma interval.
For discrete events (DLCs 1.4,1.5,6.1,6.3), the average load of the time serie is plotted. The error bar shows the max load measured in the simulation, 
which is also the load used for design. Note that the error bar is shown/assumed as symmetric even though we do not actually measure the min load in the 
time series.

# ======================================================
"""

# Root files of the various DLC analyses
fname_analysis_options_struct = [
    "results_DLCs_1p1_1p3" + os.sep + "analysis_options_struct_withUnsteadyLoads.yaml",
    "results_DLCs_1p4" + os.sep + "analysis_options_struct_withUnsteadyLoads.yaml",
    "results_DLCs_1p5" + os.sep + "analysis_options_struct_withUnsteadyLoads.yaml",
    "results_DLCs_6p1_6p3" + os.sep + "analysis_options_struct_withUnsteadyLoads.yaml",
]

# For each DLC analysis, which DLCs were simulated (in order)
list_toplot = [
    [1.1,1.3],
    [1.4],
    [1.5],
    [6.1,6.3],
]

# Choose what you want to plot. 'fac' can be used as a multiplier to what gets plotted.
qty = "MLy"
fac = -1000
# qty = "MLx"
# fac = 1000


# labs = ["Fn [N/m]","Ft [N/m]","MLx [kNm]","MLy [kNm]","FLz [kN]"]
# legs = [r"$F_n \, [N/m]$",r"$F_t \, [N/m]$","MLx [kNm]","MLy [kNm]","FLz [kN]"]


pltSize = (10, 5)

fs = 14
ls = 12



# ========================================


f1,ax1 = plt.subplots(nrows=1, ncols=1, figsize=pltSize)

ax1.tick_params(labelsize=ls)

ax1.set_ylabel("load [kNm]",fontsize=fs)
ax1.set_xlabel("U [m/s]",fontsize=fs)

for file,toplot in zip(fname_analysis_options_struct,list_toplot):

    schema = load_yaml(file)

    for key in toplot:

        nsu = len(schema["extreme"][key]['U']) * len(schema["extreme"][key]['Seeds'])
        
        U = []
        for j in range(len(schema["extreme"][key]['U'])):
            for s in range(len(schema["extreme"][key]['Seeds'])):
                U.append( float( schema["extreme"][key]['U'][j] ) )
        
        L = []
        for j in range(nsu):
            L.append(schema["extreme"][key][qty][j][0] / fac)

        ss1 = ax1.plot(U,L,'x' )
        c1 = ss1[0].get_color()
        # ax1.plot(EXTR_life_B1[i,k] , 0, 'x' , color=c1)
    
        if qty+"_avg" in schema["extreme"][key]:
            Av = []
            for j in range(nsu):
                Av.append(schema["extreme"][key][qty+"_avg"][j][0])

            St = []
            for j in range(nsu):
                St.append(schema["extreme"][key][qty+"_std"][j][0])

            print(Av)

            # ss1 = ax1.plot(U,Av,'o', color=c1 )
            ss1 = ax1.errorbar(U,Av,yerr=St, fmt='o', color=c1 )
        
        
    
f1.tight_layout()

# f1.savefig(f"{folder}/figs/MAP_{labs[k].split(' ')[0]}_{distr[k]}.eps")

plt.show()

# pltSize = (6, 3)
# fs = 20
# ls = 15

# nx=np.size(EXTR_life_B1,axis=0)
# locs = np.linspace(0.,1.,nx)

# for k in range(5):
#     f1,ax1 = plt.subplots(nrows=1, ncols=1, figsize=pltSize)
#     ax1.tick_params(labelsize=ls)

#     plt.plot(locs,EXTR_life_B1[:,k], label="EXTR")
    
#     plt.ylabel(labs[k],fontsize=fs)
#     plt.xlabel(r"$r/R$",fontsize=fs)
#     # plt.legend()

#     plt.tight_layout()
#     plt.savefig(f"{folder}/figs/{labs[k].split(' ')[0]}_{distr[k]}.eps")
# plt.show()