import os
import yaml, copy

import sys, shutil
import numpy as np

from wisdem.inputs import load_yaml
import matplotlib.pyplot as plt

from XtrFat.XtrFat import my_write_yaml

#==================== DEFINITIONS  =====================================

## File management
wt_base_folder = os.path.dirname(os.path.realpath(__file__))  # get path to this file

# The following comes from the full OpenFAST simulation
wt_base_folder = "/Users/dcaprace/Library/CloudStorage/OneDrive-UCL/2023_AIAA_ComFi/results/5_openfastRun/results_11vels_6seeds/"

# The following comes from the compute_tilde_loads
tileLoadFile = "Tilde_loads_LstSqr.yaml" 
output_subfolder = "3vels_120s_10yrExtr" #for plots and output files
output_subfolder = "3vels_300s_1yrExtr" 
output_subfolder = "11vels_600s_fatOnly" 
tileLoadFile = f"/Users/dcaprace/Library/CloudStorage/OneDrive-UCL/2023_AIAA_ComFi/results/4_compareConstraints/{output_subfolder}/Tilde_loads_LstSqr.yaml" 


# If the following is true, we add the tilde loads to the full OpenFAST output schema 
# so the tilde loads can be used in subsequent optimizations.
edit_schema_with_validated_tilde_loads = True

leg_loc = ["U","L"] # location. Could add "TE"
nx = 30 #number of spanwise stations. Could get it fron the file but it's just easier to provide it

leg_src = ["DEL","extreme"] # source of the strain used in the tilde load computation. "DEL"=fatigue, "extreme"=extreme


showAllTheSame = False

# ============== inits ===========================

Sult = 3500.e-6 #HARDCODED. Should come from something like analysis_opts["constraints"]["blade"]["strains_spar_cap_ss"]["max"]

nf = 1
n_span = nx
n_locs = len(leg_loc) #number of location around the section that constrained in the optimization, and for which we want to evaluate tilde loads
n_src = len(leg_src)

# Obtain the equivalent load corresponding to the equivalent strain in spars
#    We can use the factors y/EI11, x/EI22, 1/EA, already computed for the combili factors. 

ooEA = np.zeros((nf,n_span))
yoEIxx = np.zeros((nf,n_span,n_locs))
xoEIyy = np.zeros((nf,n_span,n_locs))



strain = np.zeros((nf,n_span,n_locs,n_src)) # 2 for U

DEMx = np.zeros((nf,n_span,n_src))
DEMy = np.zeros((nf,n_span,n_src))
DEFz = np.zeros((nf,n_span,n_src))


# ----------------------------------------------------------------------------------------------
#    specific preprocessing 

ext = ''

# filling our data structures with data from the yaml
for ifo in [0]:

    folder_arch = wt_base_folder

    simfolder = folder_arch + os.sep + 'sim' + os.sep + 'iter_0'


    schema = load_yaml( folder_arch + os.sep + "analysis_options_struct_withDEL.yaml")
    combili_channels = load_yaml( simfolder + os.sep + 'extra' + os.sep + 'combili_channels.yaml')

    if not os.path.isdir(os.path.join(folder_arch, output_subfolder)):
        os.system(f"mkdir {os.path.join(folder_arch, output_subfolder)}")

    # checks
    if nx != len(schema['DEL']['grid_nd']):
        raise ValueError("incorrect nx specified")
    if n_span != combili_channels["n_span"]:
        raise ValueError("incorrect nx specified")
    
    for isrc,src in enumerate(leg_src):
        if not src in schema:
            print(f"CAUTION: I did not find {src} in the output yaml you provided for {folder_arch}.")
            print("          I will just skip it.")
            leg_src.pop(isrc)

    combili_channels.pop("n_span")
    
    for i in range(n_span):

        ooEA[ifo,i]  = combili_channels["BladeSparU_Strain_Stn%d"%(i+1) ]["B1N0%02dFLz"%(i+1)] *1e-3

        for iloc,loc in enumerate(leg_loc):
            yoEIxx[ifo,i,iloc] = combili_channels["BladeSpar%s_Strain_Stn%d"%(loc,i+1) ]["B1N0%02dMLx"%(i+1)] *1e-3
            xoEIyy[ifo,i,iloc] = combili_channels["BladeSpar%s_Strain_Stn%d"%(loc,i+1) ]["B1N0%02dMLy"%(i+1)] *1e-3
            for isrc,src in enumerate(leg_src):
                strain[ifo,i,iloc,isrc] = schema[src]["StrainSpar%s"%(loc)][i]
        
        for isrc,src in enumerate(leg_src):            
            DEMx[ifo,i,isrc] = schema[src]["deMLx"][i]
            DEMy[ifo,i,isrc] = schema[src]["deMLy"][i]
            DEFz[ifo,i,isrc] = schema[src]["deFLz"][i]

# ----------------------------------------------------------------------------------------------
#    Read the yaml 

schema_TildeLoads = load_yaml(tileLoadFile)

TildeMx_perStrain = np.zeros((n_span,n_locs,n_src))
TildeMy_perStrain = np.zeros((n_span,n_locs,n_src))
TildeFz_perStrain = np.zeros((n_span,n_locs,n_src))

if "DEL" in leg_src:
    isrc = leg_src.index("DEL")

    if "DEL_Tilde_ss" in schema_TildeLoads:
        iloc = leg_loc.index("U")
        TildeMx_perStrain[:,iloc,isrc] = schema_TildeLoads["DEL_Tilde_ss"]["deMLxPerStrain"]
        TildeMy_perStrain[:,iloc,isrc] = schema_TildeLoads["DEL_Tilde_ss"]["deMLyPerStrain"]
        TildeFz_perStrain[:,iloc,isrc] = schema_TildeLoads["DEL_Tilde_ss"]["deFLzPerStrain"]

        iloc = leg_loc.index("L")
        TildeMx_perStrain[:,iloc,isrc] = schema_TildeLoads["DEL_Tilde_ps"]["deMLxPerStrain"]
        TildeMy_perStrain[:,iloc,isrc] = schema_TildeLoads["DEL_Tilde_ps"]["deMLyPerStrain"]
        TildeFz_perStrain[:,iloc,isrc] = schema_TildeLoads["DEL_Tilde_ps"]["deFLzPerStrain"]
    else: 
        print(f"CAUTION: DEL exist in your simulation but not in the tilde file. I will skip it.")
        leg_src.pop(isrc)

if "extreme" in leg_src:
    isrc = leg_src.index("extreme")

    if "EXTR_Tilde_ss" in schema_TildeLoads:
        iloc = leg_loc.index("U")
        TildeMx_perStrain[:,iloc,isrc] = schema_TildeLoads["EXTR_Tilde_ss"]["deMLxPerStrain"]
        TildeMy_perStrain[:,iloc,isrc] = schema_TildeLoads["EXTR_Tilde_ss"]["deMLyPerStrain"]
        TildeFz_perStrain[:,iloc,isrc] = schema_TildeLoads["EXTR_Tilde_ss"]["deFLzPerStrain"]

        iloc = leg_loc.index("L")
        TildeMx_perStrain[:,iloc,isrc] = schema_TildeLoads["EXTR_Tilde_ps"]["deMLxPerStrain"]
        TildeMy_perStrain[:,iloc,isrc] = schema_TildeLoads["EXTR_Tilde_ps"]["deMLyPerStrain"]
        TildeFz_perStrain[:,iloc,isrc] = schema_TildeLoads["EXTR_Tilde_ps"]["deFLzPerStrain"]
    
    else: 
        print(f"CAUTION: extreme exist in your simulation but not in the tilde file. I will skip it.")
        leg_src.pop(isrc)

TildeMx = np.zeros((n_span,n_locs,n_src))
TildeMy = np.zeros((n_span,n_locs,n_src))
TildeFz = np.zeros((n_span,n_locs,n_src))

for iloc,loc in enumerate(leg_loc):
    for isrc,src in enumerate(leg_src):
        TildeMx[:,iloc,isrc] = TildeMx_perStrain[:,iloc,isrc] * strain[0,:,iloc,isrc]
        TildeMy[:,iloc,isrc] = TildeMy_perStrain[:,iloc,isrc] * strain[0,:,iloc,isrc]
        TildeFz[:,iloc,isrc] = TildeFz_perStrain[:,iloc,isrc] * strain[0,:,iloc,isrc]



# ----------------------------------------------------------------------------------------------
# -- Final plots
locs = np.linspace(0.,1.,nx)



def strainFormula(iloc,Mx,My,Fz):
    return (Mx * yoEIxx[0,:,iloc] + My * xoEIyy[0,:,iloc] + Fz * ooEA[0,:])

# def strainFormula_NOF(iloc,Mx,My,Fz):
#     return (Mx * yoEIxx[0,:,iloc] + My * xoEIyy[0,:,iloc] - Fz * ooEA[0,:])

ptrn = ['-','--']

# -Strains-

for iloc,loc in enumerate(leg_loc):
    plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

    for isrc,src in enumerate(leg_src):
        plt.plot(locs,strain[0,:,iloc,isrc], ptrn[isrc] , label=f"actual {src}", color='k')
        plt.plot(locs,strainFormula(iloc,DEMx[0,:,isrc],DEMy[0,:,isrc],DEFz[0,:,isrc]), ptrn[isrc], label="from DE") 
        plt.plot(locs,strainFormula(iloc,TildeMx[:,iloc,isrc],TildeMy[:,iloc,isrc],TildeFz[:,iloc,isrc]), ptrn[isrc], label=f"tilde {src}")

    plt.plot(locs[[0,-1]],[ Sult]*2,':k')
    plt.plot(locs[[0,-1]],[-Sult]*2,':k')

    plt.ylabel(f"strain{loc}")
    plt.xlabel("r/R")
    plt.legend()
    plt.savefig(os.path.join(wt_base_folder, output_subfolder,f"strain{loc}.png"))


# LOADS
for iloc,loc in enumerate(leg_loc):
    fig,ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 5))
    
    for isrc,src in enumerate(leg_src):
        #WHEN WE CONSIDER THAT ALL THINGS HAVE THE SAME WEIGHT
        SimpleTilde = strain[0,:,iloc,isrc] / (yoEIxx[0,:,iloc] + xoEIyy[0,:,iloc] + ooEA[0,:])  #scaling the strain to the proper units

        
        ax[0].plot(locs, DEMx[0,:,isrc], ptrn[isrc], label=f"{src}", )
        ax[1].plot(locs, DEMy[0,:,isrc], ptrn[isrc], label=f"{src}")
        ax[2].plot(locs, DEFz[0,:,isrc], ptrn[isrc], label=f"{src}")

        ax[0].plot(locs, TildeMx[:,iloc,isrc], ptrn[isrc], label=f"tilde {src}")
        ax[1].plot(locs, TildeMy[:,iloc,isrc], ptrn[isrc], label=f"tilde {src}")
        ax[2].plot(locs, TildeFz[:,iloc,isrc], ptrn[isrc], label=f"tilde {src}")
        
        if showAllTheSame:
            ax[0].plot(locs, SimpleTilde, ptrn[isrc], label=f"all the same")
            ax[1].plot(locs, SimpleTilde, ptrn[isrc], label=f"all the same")
            ax[2].plot(locs, SimpleTilde, ptrn[isrc], label=f"all the same")

    # plt.plot(locs, TildeMxL , label="tilde L")
    # plt.plot(locs, TildeMxU , label="tilde U")
    ax[0].set_ylabel('Mx')
    ax[1].set_ylabel('My')
    ax[2].set_ylabel('Fz')
    ax[2].set_xlabel("r/R")
    ax[0].set_title(loc)
    plt.legend()
    plt.savefig(os.path.join(wt_base_folder, output_subfolder,f"loads_{iloc}.png"))


plt.show()

# ----------------------------------------------------------------------------------------------
# -- Final plots

if edit_schema_with_validated_tilde_loads:
    print(f"Now editing {folder_arch}/analysis_options_struct_withDEL.yaml ...")

    if "DEL" in leg_src:
        isrc = leg_src.index("DEL")

        iloc = leg_loc.index("L")
        schema["DEL_Tilde_ps"] = {}
        schema["DEL_Tilde_ps"]["deMLx"] = TildeMx[:,iloc,isrc].tolist()
        schema["DEL_Tilde_ps"]["deMLy"] = TildeMy[:,iloc,isrc].tolist()
        schema["DEL_Tilde_ps"]["deFLz"] = TildeFz[:,iloc,isrc].tolist()

        iloc = leg_loc.index("U")
        schema["DEL_Tilde_ss"] = {}
        schema["DEL_Tilde_ss"]["deMLx"] = TildeMx[:,iloc,isrc].tolist()
        schema["DEL_Tilde_ss"]["deMLy"] = TildeMy[:,iloc,isrc].tolist()
        schema["DEL_Tilde_ss"]["deFLz"] = TildeFz[:,iloc,isrc].tolist()

        # EXPORT FOR MACH
        fname_fatMach = os.path.join(wt_base_folder, output_subfolder, "fatigueTildeLoads_forMACH.yaml")
        fat_schema = {}
        fat_schema["DEL_Tilde_ss"] = schema["DEL_Tilde_ss"]
        fat_schema["DEL_Tilde_ps"] = schema["DEL_Tilde_ps"]
        my_write_yaml(fat_schema, fname_fatMach)

    if "extreme" in leg_src:
        isrc = leg_src.index("extreme")

        iloc = leg_loc.index("L")
        schema["EXTR_Tilde_ps"] = {}
        schema["EXTR_Tilde_ps"]["deMLx"] = TildeMx[:,iloc,isrc].tolist()
        schema["EXTR_Tilde_ps"]["deMLy"] = TildeMy[:,iloc,isrc].tolist()
        schema["EXTR_Tilde_ps"]["deFLz"] = TildeFz[:,iloc,isrc].tolist()

        iloc = leg_loc.index("U")
        schema["EXTR_Tilde_ss"] = {}
        schema["EXTR_Tilde_ss"]["deMLx"] = TildeMx[:,iloc,isrc].tolist()
        schema["EXTR_Tilde_ss"]["deMLy"] = TildeMy[:,iloc,isrc].tolist()
        schema["EXTR_Tilde_ss"]["deFLz"] = TildeFz[:,iloc,isrc].tolist()
        

        # EXPORT FOR MACH
        fname_extrMach = os.path.join(wt_base_folder, output_subfolder, "extrTildeLoads_forMACH.yaml")
        extr_schema = {}
        extr_schema["EXTR_Tilde_ss"] = schema["EXTR_Tilde_ss"]
        extr_schema["EXTR_Tilde_ps"] = schema["EXTR_Tilde_ps"]
        my_write_yaml(extr_schema, fname_extrMach)
    
    fname_output = os.path.join(wt_base_folder, output_subfolder, "analysis_options_struct_withDEL.yaml")
    my_write_yaml(schema, fname_output)