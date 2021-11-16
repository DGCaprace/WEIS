import os
import ast
from pyoptsparse.pyOpt_history import History

# import sys, shutil
import numpy as np
import matplotlib.pyplot as plt


#==================== DEFINITIONS  =====================================

""" This scripts take a hst file, and the nominal output of DV and DVGroups for the same structure to
regenerate a .dat file that has the DV values from the hst.

"""

## File management
mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file


DV_input = "generic_DVCentres.dat"
DVGroup_input = "generic_DVGroupCentres.dat"
HST_input = "/Users/dg/Documents/BYU/simulation_data/ATLANTIS/MDAO/Aerostructural/fromMarco/struct_sized.hst" #preoptimized structure
DV_output = "/Users/dg/Documents/BYU/simulation_data/ATLANTIS/MDAO/Aerostructural/fromMarco/struct_sized_DVCentres.dat"

# HST_input = "/Users/dg/Documents/BYU/simulation_data/ATLANTIS/MDAO/Aerostructural/Optimization/1pt_fatigue_44949859_L3/Opt_output/SLSQP_hist_L31pt_fatigue_44949859.hst"

doPlot = True

#==================== LOAD HiFi DVs DATA =====================================

DV_file = DV_input
ncon = 0

#Read the constitutive component file
with open(DV_input, 'r') as f:
    lines = f.readlines()

    nentry = len(lines[0].split(" "))
    ncon = nentry-6

    HiFiDVs_idx  = np.zeros(len(lines), int)
    HiFiDVs_pos = np.zeros((len(lines),3))
    HiFiDVs_thi = np.zeros(len(lines))
    HiFiDVs_con = np.zeros((len(lines), ncon))
    HiFiDVs_des = [None]*len(lines)

    i = 0
    for line in lines:
        buff = line.split(" ")
        HiFiDVs_idx[i] = buff[0]
        HiFiDVs_pos[i,:] = [ float(b) for b in buff[1:4] ]
        # HiFiDVs_thi[i] = float(buff[4]) 
        # HiFiDVs_con[i,:] = [ float(b) for b in buff[5:-1] ] 
        HiFiDVs_des[i] = buff[-1][:-1] #discarding the '\n'
        i+=1

#Read the constitutive component file
with open(DVGroup_input, 'r') as f:
    lines = f.readlines()

    # nentry = len(lines[0].split(" "))
    # ncon = nentry-6

    HiFiGrp_idx  = np.zeros(len(lines), int)
    # HiFiGrp_pos = np.zeros((len(lines),3))
    # HiFiGrp_thi = np.zeros(len(lines))
    # HiFiGrp_con = np.zeros((len(lines), ncon))
    # HiFiGrp_des = [None]*len(lines)
    HiFiGrp_icmp = [None]*len(lines)

    i = 0
    for line in lines:
        buff = line.split(" ")
        HiFiGrp_idx[i] = buff[0]
        # HiFiGrp_pos[i,:] = [ float(b) for b in buff[1:4] ]
        # HiFiGrp_thi[i] = float(buff[4]) 
        # HiFiGrp_con[i,:] = [ float(b) for b in buff[5:-1] ] 
        buff = line.split('"')
        # HiFiGrp_des[i] = buff[1]
        HiFiGrp_icmp[i] = buff[3]
        i+=1


#Allocate a mapping object: 
#  binds each group number to the list of constitutive elements it covers
HiFiDVs_mapping = [[] for i in range(len(HiFiGrp_idx))]

for i in range(len(HiFiGrp_idx)):
    tmp = ",".join(HiFiGrp_icmp[i][1:-1].split())
    icmp = ast.literal_eval('['+tmp+']')
    HiFiDVs_mapping[HiFiGrp_idx[i]] = icmp

print(HiFiDVs_mapping)
#TODO: check there in not twice the same index in the mapping

#==================== LOAD HHST DATA =====================================
optHist = History(HST_input)
histValues = optHist.getValues()

print(histValues.keys())

DVs_hst = histValues["struct"][-1] #end value of the optim

#UPDATE THE THICKNESSES
for i in range(len(HiFiDVs_mapping)):

    for ic in HiFiDVs_mapping[i]:
        # find the line in the DV file that corresponds to the component index ic
        line = np.where(HiFiDVs_idx==ic)[0]
        # update the value of the thickness there
        HiFiDVs_thi[line] = DVs_hst[i]

#==================== Export to file =====================================

#Read the constitutive component file
with open(DV_output, 'w') as f:
    for i in range(len(HiFiDVs_idx)):
        pattern = "%i %6.5e %6.5e %6.5e %6.5e "
        # pattern += "%6.5e "*ncon
        pattern += "%s\n"

        f.write(pattern%(
            HiFiDVs_idx[i],
            HiFiDVs_pos[i,0], HiFiDVs_pos[i,1], HiFiDVs_pos[i,2],
            HiFiDVs_thi[i],
            # tuple(HiFiDVs_con[i,:]),
            HiFiDVs_des[i]
        ))





# #==================== Compare DV Plots #====================

# #------- skin ----------
# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 5))

# for isk in range(len(skinLoFi)):
    
#     values = np.zeros(len(ylf_skn_oR))
#     for j in range(nhf_web):
#         values[2*j] = skin_hifi[j,isk,1]
#         values[2*j+1] = skin_hifi[j,isk,1]

#     hp = ax.plot(ylf_skn_oR,values, '-', label=skinLoFi[isk])
        
# ax.set_ylabel("thickness [mm]")
# ax.set_xlabel("r/R")
# plt.legend()


# #------- webs ----------
# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 5))

# for isk in range(len(websLoFi)):
    
#     values = np.zeros(len(ylf_web_oR))
#     for j in range(nhf_web):
#         #the thickness is 0 where the web is not defined.
#         values[2*j] = max(webs_hifi[j,isk],0.0)
#         values[2*j+1] = max(webs_hifi[j,isk],0.0)
        
#     hp = ax.plot(ylf_web_oR,values, '-', label=websLoFi[isk])
        
# ax.set_ylabel("thickness [mm]")
# ax.set_xlabel("r/R")
# plt.legend()



plt.show()