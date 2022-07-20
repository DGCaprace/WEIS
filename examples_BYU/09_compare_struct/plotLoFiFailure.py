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
USE_LTILDE = False #use the tilde values instead of the directly aggregated Mx,My,Fz

m_wohler = 10
n_life_eq = 1
eta = 1.35
R = 89.166
R0 = 2.8

runWISDEM = False
plotActualDamage = False
withLegend = True

#-- plot params--
figsize=(10, 4)
fs=15
plotTight = True

#==================== DEFINITIONS  =====================================

## File management
mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
fname_modeling_options = mydir + os.sep + "modeling_options.yaml"
fname_analysis_options = mydir + os.sep + "analysis_options_struct.yaml"

# #Original constant thickness model, under nominal loads
withNominal = True
fname_wt_input = mydir + os.sep + "Madsen2019_10_forWEIS_isotropic_IC.yaml"
fname_loads = mydir + os.sep + "Madsen2019_10_forWEIS_isotropic_IC/nominalLoads.yaml" #the loading condition you want to use (should have a nominal section)
folder_arch = mydir + os.sep + "LoFiEval_isotropic_nominalLoads"
importHifiCstr = "Madsen2019_10_forWEIS_isotropic_IC/hifiCstr_nominalg.npz"
importHifiCstrCsv = []  #a csv file output from Paraview using a slice of the TACS solution file (f5/plt)
HFc = 0 # index of the corresponding constraint in HiFi


#==================== ======== =====================================
## Preprocessing: filling in the loads and relevant parameters

# TODO: replace all this by just reading the strain in the output yaml from the 09/driver

if not os.path.isdir(folder_arch):
    runWISDEM = True

if withEXTR and withDEL and withNominal or withDEL and withNominal:
    raise ValueError("choose one of EXTRM, DEL or norminal")

if withEXTR and withNominal:
    raise ValueError("can't do withEXTR and withNominal: nominal loads actually replace the extreme loads")



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
    schema["extreme"]["deMLx"] = schema_loads[src_name]["mean"]["deMLx"]
    schema["extreme"]["deMLy"] = schema_loads[src_name]["mean"]["deMLy"]
    schema["extreme"]["deFLz"] = schema_loads[src_name]["mean"]["deFLz"]
    schema["constraints"]["blade"]["extreme_loads_from_user_inputs"] = True #we are using that channel as a way to specify a loading. The we will read the corresponding strain the EXTRM strain  output



if withDEL:
    schema["DEL"] = {}
    schema["DEL"]["description"] = schema_loads["DEL"]["description"]
    schema["DEL"]["grid_nd"] = schema_loads["DEL"]["grid_nd"]
    if USE_LTILDE:
        schema["DEL"]["deMLx"] = schema_loads["DEL"]["mean"]["deMLxTilde"]
        schema["DEL"]["deMLy"] = schema_loads["DEL"]["mean"]["deMLyTilde"]
        schema["DEL"]["deFLz"] = schema_loads["DEL"]["mean"]["deFLzTilde"]
    else:
        schema["DEL"]["deMLx"] = schema_loads["DEL"]["mean"]["deMLx"]
        schema["DEL"]["deMLy"] = schema_loads["DEL"]["mean"]["deMLy"]
        schema["DEL"]["deFLz"] = schema_loads["DEL"]["mean"]["deFLz"]

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

exp = 1.0
if withDEL and plotActualDamage:
    exp = m_wohler

max_strain = schema["constraints"]["blade"]["strains_spar_cap_ss"]["max"]

WISDEMout = folder_arch + "/blade_out.npz"

fig3, ax3 = plt.subplots(nrows=1, ncols=1, figsize=figsize)
fig4, ax4 = plt.subplots(nrows=1, ncols=1, figsize=figsize)

with np.load(WISDEMout) as a:

    r = np.array(a["rotorse.rs.z_az_m"])
    r = (r-r[0])/(r[-1]-r[0])
    
    if withNominal or withEXTR:
        #My passed loading stuff:
        data3 = a["rotorse.rs.extreme_strains.F3_N"]  #should be = to my input
        data4 = a["rotorse.rs.extreme_strains.M1_N*m"]  #should be = to my input (well, my input but set in principal axes)
        data5 = a["rotorse.rs.extreme_strains.M2_N*m"]  #should be = to my input (well, my input but set in principal axes)
        data1 = eta * a["rotorse.rs.extreme_strains.strainU_spar"] / max_strain #rebuild the failure constraint
        data2 = eta * a["rotorse.rs.extreme_strains.strainL_spar"] / max_strain #rebuild the failure constraint

        # data1 = a["rotorse.xu_strain_spar"]
        # data2 = a["rotorse.yu_strain_spar"]
        # data1 = a["rotorse.xl_strain_spar"]
        # data2 = a["rotorse.yl_strain_spar"]

    if withDEL:
        #My passed loading stuff:
        data3 = a["rotorse.rs.fatigue_strains.F3_N"]  #should be = to my input
        data4 = a["rotorse.rs.fatigue_strains.M1_N*m"]  #should be = to my input (well, my input but set in principal axes)
        data5 = a["rotorse.rs.fatigue_strains.M2_N*m"]  #should be = to my input (well, my input but set in principal axes)
        data1 = (eta * a["rotorse.rs.fatigue_strains.strainU_spar"] / max_strain)**exp #rebuild the damage constraint
        data2 = (eta * a["rotorse.rs.fatigue_strains.strainL_spar"] / max_strain)**exp #rebuild the damage constraint
    
    # #original gust stuff:
    # data = a["rotorse.rs.strains.F3_N"]
    # data = a["rotorse.rs.strains.M1_N*m"]
    # data = a["rotorse.rs.strains.M2_N*m"]
    # data1 = a["rotorse.rs.strains.strainU_spar"] / max_strain
    # data2 = a["rotorse.rs.strains.strainL_spar"] / max_strain

    # r = range(4)
    # data = a["rotorse.rs.constr.constr_max_strainU_spar"]
    # data = a["rotorse.rs.constr.constr_max_strainL_spar"]

    hp1 = ax3.plot(r, data2,'o-', label=f'PS') #NEGATIVE to make them both positive??? but the strain should be > 0 on the PS!!??
    hp2 = ax3.plot(r,-data1,'o-', label=f'SS')  

    ax4.plot(r,data3,'-', label=f'F3')
    ax4.plot(r,data4,'-', label=f'M1')
    ax4.plot(r,data5,'-', label=f'M2')

colors = [hp1[0].get_color(), hp2[0].get_color()]


if importHifiCstr:
    f= np.load(mydir + os.sep + importHifiCstr)

    ylf_skn_oR = f["ylf_skn_oR"]
    skin_hifi_con = f["skin_hifi_con"]
    nhf_skn = f["nhf_skn"]
    ncon = f["ncon"]
    spars = f["spars"].tolist()
    spars_legend = f["spars_legend"]
    skinLoFi = f["skinLoFi"]

    if ncon>0:
        #------- skin ----------
        # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 5))
        
        for isk in range(len(skinLoFi)):
            
            values = np.zeros((len(ylf_skn_oR),ncon))
            for c in range(ncon):
                for j in range(nhf_skn):
                    values[2*j,c] = (eta * skin_hifi_con[j,isk,c])**exp
                    values[2*j+1,c] = (eta * skin_hifi_con[j,isk,c])**exp

            # hp = ax.plot(ylf_skn_oR,values[:,0], '-', label=skinLoFi[isk])
            # if ncon>1:
            #     ax.plot(ylf_skn_oR,values[:,1], '--', color=hp[0].get_color())

            if len(spars)>0:
                if any( [ skinLoFi[isk] in sp for sp in spars ]):
                    isp = spars.index(skinLoFi[isk]) 
                    hp = ax3.plot(ylf_skn_oR,values[:,HFc], 'x--', label=spars_legend[isp], color=colors[isp])

for file in importHifiCstrCsv:
    # import csv
    # with open(importHifiCstrCsv, newline='') as csvfile:
    #     reader = csv.reader(csvfile, delimiter=',', quotechar='#')
    #     for row in reader:
    #         print(row)

    data = np.genfromtxt(file, delimiter=',')
    sortID = np.argsort(data[:,3]) #along y

    mask_ss = np.where( data[sortID,2]>0 ) 
    mask_ps = np.where( data[sortID,2]<0 ) 
    

    #  2347 #suction side is where x>0
        
    #SS
    yspar = (data[sortID[mask_ss],3] - R0) / (R-R0) 
    lamspar = (data[sortID[mask_ss],7] *eta)**exp
    ax3.plot(yspar,lamspar, '--', label='ss', color=colors[1])

    #PS
    yspar = (data[sortID[mask_ps],3] - R0) / (R-R0)
    lamspar = (data[sortID[mask_ps],7] * eta)**exp
    ax3.plot(yspar,lamspar, '--', label='ss', color=colors[0])



ax3.set_xlabel(r"$r/R$",fontsize=fs)
ax4.set_xlabel(r"$r/R$",fontsize=fs)

ax4.set_ylabel(r"loads",fontsize=fs)

if withLegend:
    ax3.legend()
    ax4.legend()

if withDEL:
    if plotActualDamage:
        ax3.set_ylabel(r"$D$",fontsize=fs)
    else:
        ax3.set_ylabel(r"$D^{1/m}$",fontsize=fs)
    fig3.savefig(folder_arch + f"/damage.eps")
    fig4.savefig(folder_arch + f"/load_fatigue.eps")
elif withNominal:
    ax3.set_ylabel(r"$Y$",fontsize=fs)
    fig3.savefig(folder_arch + f"/failure_nominal.eps")
    fig4.savefig(folder_arch + f"/load_nominal.eps")
else:
    ax3.set_ylabel(r"$Y$",fontsize=fs)
    fig3.savefig(folder_arch + f"/failure_extreme.eps")
    fig4.savefig(folder_arch + f"/load_extreme.eps")

if plotTight:
    fig3.tight_layout()
    fig4.tight_layout()


plt.show()