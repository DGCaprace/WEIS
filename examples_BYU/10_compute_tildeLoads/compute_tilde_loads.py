import os
import yaml, copy

import sys, shutil
import numpy as np

from wisdem.inputs import load_yaml
import matplotlib.pyplot as plt

from XtrFat.XtrFat import my_write_yaml

#==================== DEFINITIONS  =====================================

## File management
mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file

wt_base_name = "Madsen2019_composite_v02_IC"

mydir = '/Users/dcaprace/Library/CloudStorage/OneDrive-UCL/2023_AIAA_ComFi/results/4_compareConstraints'
mydir = '/Users/dcaprace/Library/CloudStorage/OneDrive-UCL/2023_AIAA_ComFi/results/4_compareConstraints/3vels_120s_10yrExtr'
mydir = '/Users/dcaprace/Library/CloudStorage/OneDrive-UCL/2023_AIAA_ComFi/results/4_compareConstraints/3vels_300s_1yrExtr'
# mydir = '/Users/dcaprace/Library/CloudStorage/OneDrive-UCL/2023_AIAA_ComFi/results/4_compareConstraints/11vels_600s_fatOnly'
wt_base_name = "Madsen2019_composite_v02_originalThickness"


ext_list = ["","p01","p02","p03","p04"]


leg_loc = ["U","L"] # location. Could add "TE"
nx = 30 #number of spanwise stations. Could get it fron the file but it's just easier to provide it

leg_src = ["DEL","extreme"] # source of the strain used in the tilde load computation. "DEL"=fatigue, "extreme"=extreme


showAllTheSame = False

# ============== inits ===========================

# Sult = 3500.e-6 #HARDCODED. Should come from something like analysis_opts["constraints"]["blade"]["strains_spar_cap_ss"]["max"]

nf = len(ext_list)                        
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

# filling our data structures with data from the yaml
for ifo,ext in enumerate(ext_list): 

    folder_arch = os.path.join(mydir, wt_base_name + ext)

    simfolder = folder_arch + os.sep + 'sim' + os.sep + 'iter_0'


    schema = load_yaml( folder_arch + os.sep + 'analysis_options_struct_withDEL.yaml')
    combili_channels = load_yaml( simfolder + os.sep + 'extra' + os.sep + 'combili_channels.yaml')

    # checks
    if nx != len(schema['DEL']['grid_nd']):
        raise ValueError("incorrect nx specified")
    if n_span != combili_channels["n_span"]:
        raise ValueError("incorrect nx specified")

    combili_channels.pop("n_span")
    
    for isrc,src in enumerate(leg_src):
        if not src in schema:
            print(f"CAUTION: I did not find {src} in the output yaml you provided for {wt_base_name + ext}.")
            print("          I will just skip it.")
            leg_src.pop(isrc)

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
#    solve the system            

# # d.(option1) -- KETPS HERE FOR THE RECORD, DID NOT WORK WELL WHEN IN XtrFat.py
# # Find the equivalent Mx,My,Fz that will give the same strain as the Damage-equivalent Life strain,
# #  and that also have the same ratios as DEMx,DEMy,DEFz (that is, the damage-eq loads not based on strain)
# Ltilde_life_B1[:,2] = DEL_life_B1[:,5] / (DEL_life_B1[:,2]/DEL_life_B1[:,4] * yoEIxxU + DEL_life_B1[:,3]/DEL_life_B1[:,4] * xoEIyyU + ooEA )
# Ltilde_life_B1[:,0] = DEL_life_B1[:,2]/DEL_life_B1[:,4] * Ltilde_life_B1[:,2]
# Ltilde_life_B1[:,1] = DEL_life_B1[:,3]/DEL_life_B1[:,4] * Ltilde_life_B1[:,2]

# Ltilde_life_B1[:,5] = DEL_life_B1[:,6] / (DEL_life_B1[:,2]/DEL_life_B1[:,4] * yoEIxxL + DEL_life_B1[:,3]/DEL_life_B1[:,4] * xoEIyyL + ooEA )
# Ltilde_life_B1[:,3] = DEL_life_B1[:,2]/DEL_life_B1[:,4] * Ltilde_life_B1[:,5]
# Ltilde_life_B1[:,4] = DEL_life_B1[:,3]/DEL_life_B1[:,4] * Ltilde_life_B1[:,5]

#--
# # d.(option2) -- KETPS HERE FOR THE RECORD, DID NOT WORK WELL WHEN IN XtrFat.py
# # Find the unique equivalent Mx,My,Fz that will give the same strain as the Damage-equivalent Life strain
# #   in the spars and at the TE simultaneously
# # NOTE: This one will not work: you can find the equivalent Mx,My,Fz that give this strain, but this is not what you want!
# #       Imagine a simple beam with pure bending cyclinc loads. The DEstrain will be the same in U and L spars, and the equivalent 
# #       loading will be pure Fz, and not bending! 
# A = np.zeros([3,3])
# for i in range(n_span):
#     A[0,0] = yoEIxxU[i]
#     A[0,1] = xoEIyyU[i]
#     A[1,0] = yoEIxxL[i]
#     A[1,1] = xoEIyyL[i]
#     A[2,0] = yoEIxxTE[i]
#     A[2,1] = xoEIyyTE[i]
#     A[:,2] = ooEA[i]
#     b = DEL_life_B1[i,5:]
#     sol = np.linalg.solve(A, b) * 1.e3  # A assumes x vector in thousands
#     Ltilde_life_B1[i,0] = sol[0] 
#     Ltilde_life_B1[i,1] = sol[1] 
#     Ltilde_life_B1[i,2] = sol[2] 
# Ltilde_life_B1[:,3] = Ltilde_life_B1[:,0]
# Ltilde_life_B1[:,4] = Ltilde_life_B1[:,1]
# Ltilde_life_B1[:,5] = Ltilde_life_B1[:,2]

# #--
# # d.(option3) -- KETPS HERE FOR THE RECORD, DID NOT WORK WELL WHEN IN XtrFat.py
# # Find the equivalent Mx,My,Fz that will give the same strain as the Damage-equivalent Life strain,
# #  and that simply have equal weights (i.e., Mx=My=Fz)
# Ltilde_life_B1[:,0] = DEL_life_B1[:,5] / (yoEIxxU + xoEIyyU + ooEA ) * 1.e3
# Ltilde_life_B1[:,1] = DEL_life_B1[:,5] / (yoEIxxU + xoEIyyU + ooEA ) * 1.e3
# Ltilde_life_B1[:,2] = DEL_life_B1[:,5] / (yoEIxxU + xoEIyyU + ooEA ) * 1.e3

# Ltilde_life_B1[:,3] = DEL_life_B1[:,6] / (yoEIxxL + xoEIyyL + ooEA ) * 1.e3
# Ltilde_life_B1[:,4] = DEL_life_B1[:,6] / (yoEIxxL + xoEIyyL + ooEA ) * 1.e3
# Ltilde_life_B1[:,5] = DEL_life_B1[:,6] / (yoEIxxL + xoEIyyL + ooEA ) * 1.e3


# d.(option4)
# Find the tilde loads that work best for a couple of different cases

def solve_tilde_loads(
        yoEIxxU, xoEIyyU, ooEA,
        strainU,
        ):

    nf = strainU.shape[0]
    n_span = strainU.shape[1]
    n_locs = strainU.shape[2]
    n_src = strainU.shape[3]

    newTildeMxU = np.zeros( (n_span,n_locs,n_src) )
    newTildeMyU = np.zeros( (n_span,n_locs,n_src) )
    newTildeFzU = np.zeros( (n_span,n_locs,n_src) )

    # least square solve if there are more perturbations than the number of unknowns, which in this case is 3
    if nf>3:
        solver = np.linalg.lstsq
        post = lambda x: x[0]
    else:
        solver = np.linalg.solve
        post = lambda x: x

    A = np.zeros([nf,3])
    # for each strain ditribution, solve the tilde load at every spanwise location
    for iloc in range(n_locs):
        for i in range(n_span):
            A[:,0] = yoEIxxU[:,i,iloc]
            A[:,1] = xoEIyyU[:,i,iloc]
            A[:,2] = ooEA[:,i]
            for isrc in range(n_src):
                b = strainU[:,i,iloc,isrc] #rhs
                sol = solver(A, b)
                sol = post(sol)
                newTildeMxU[i,iloc,isrc] = sol[0]
                newTildeMyU[i,iloc,isrc] = sol[1] 
                newTildeFzU[i,iloc,isrc] = sol[2] 

    return newTildeMxU, newTildeMyU, newTildeFzU

TildeMx, TildeMy, TildeFz = solve_tilde_loads(
        yoEIxx, xoEIyy, ooEA,
        strain,
        )

# ----------------------------------------------------------------------------------------------
# -- Final exports 

# We write the tilde loads 'per unit strain'

schema_out = {}

if "DEL" in leg_src:
    isrc = leg_src.index("DEL")

    iloc = leg_loc.index("U")
    schema_out["DEL_Tilde_ss"] = {}
    schema_out["DEL_Tilde_ss"]["deMLxPerStrain"] = (TildeMx[:,iloc,isrc] / strain[0,:,iloc,isrc]).tolist()
    schema_out["DEL_Tilde_ss"]["deMLyPerStrain"] = (TildeMy[:,iloc,isrc] / strain[0,:,iloc,isrc]).tolist()
    schema_out["DEL_Tilde_ss"]["deFLzPerStrain"] = (TildeFz[:,iloc,isrc] / strain[0,:,iloc,isrc]).tolist()

    iloc = leg_loc.index("L")
    schema_out["DEL_Tilde_ps"] = {}
    schema_out["DEL_Tilde_ps"]["deMLxPerStrain"] = (TildeMx[:,iloc,isrc] / strain[0,:,iloc,isrc]).tolist()
    schema_out["DEL_Tilde_ps"]["deMLyPerStrain"] = (TildeMy[:,iloc,isrc] / strain[0,:,iloc,isrc]).tolist()
    schema_out["DEL_Tilde_ps"]["deFLzPerStrain"] = (TildeFz[:,iloc,isrc] / strain[0,:,iloc,isrc]).tolist()

if "extreme" in leg_src:
    isrc = leg_src.index("extreme")

    iloc = leg_loc.index("U")
    schema_out["extreme_Tilde_ss"] = {}
    schema_out["extreme_Tilde_ss"]["deMLxPerStrain"] = (TildeMx[:,iloc,isrc] / strain[0,:,iloc,isrc]).tolist()
    schema_out["extreme_Tilde_ss"]["deMLyPerStrain"] = (TildeMy[:,iloc,isrc] / strain[0,:,iloc,isrc]).tolist()
    schema_out["extreme_Tilde_ss"]["deFLzPerStrain"] = (TildeFz[:,iloc,isrc] / strain[0,:,iloc,isrc]).tolist()

    iloc = leg_loc.index("L")
    schema_out["extreme_Tilde_ps"] = {}
    schema_out["extreme_Tilde_ps"]["deMLxPerStrain"] = (TildeMx[:,iloc,isrc] / strain[0,:,iloc,isrc]).tolist()
    schema_out["extreme_Tilde_ps"]["deMLyPerStrain"] = (TildeMy[:,iloc,isrc] / strain[0,:,iloc,isrc]).tolist()
    schema_out["extreme_Tilde_ps"]["deFLzPerStrain"] = (TildeFz[:,iloc,isrc] / strain[0,:,iloc,isrc]).tolist()


my_write_yaml(schema_out, os.path.join(mydir,"Tilde_loads_LstSqr.yaml") )


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
        # plt.plot(locs,-(DEMx[0,:]  *yoEIxxU[0,:]+DEMy[0,:]  *xoEIyyU[0,:]+DEFz[0,:]  *ooEA[0,:]) , label="DE-")
        plt.plot(locs,strainFormula(iloc,TildeMx[:,iloc,isrc],TildeMy[:,iloc,isrc],TildeFz[:,iloc,isrc]), ptrn[isrc], label=f"tilde {src}")

        # if src == "extreme":
        #     plt.plot(locs,strainFormula_NOF(iloc,DEMx[0,:,isrc],DEMy[0,:,isrc],DEFz[0,:,isrc]), 'x--', label="from DE, NO F") 
        
    plt.ylabel(f"strain{loc}")
    plt.xlabel("r/R")
    plt.legend()
    plt.savefig(mydir+f"/strain{loc}.png")


# -plots-


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
    plt.savefig(mydir+f"/loads_{iloc}.png")



# WEIGHTING FACTOR = LOAD PER UNIT STRAIN
for iloc,loc in enumerate(leg_loc):
    fig,ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 5))
    
    for isrc,src in enumerate(leg_src):
        #WHEN WE CONSIDER THAT ALL THINGS HAVE THE SAME WEIGHT
        SimpleTilde = 1 / (yoEIxx[0,:,iloc] + xoEIyy[0,:,iloc] + ooEA[0,:])  #scaling the strain to the proper units

        ax[0].plot(locs, TildeMx[:,iloc,isrc] / strain[0,:,iloc,isrc], ptrn[isrc], label=f"tilde {src}")
        ax[1].plot(locs, TildeMy[:,iloc,isrc] / strain[0,:,iloc,isrc], ptrn[isrc], label=f"tilde {src}")
        ax[2].plot(locs, TildeFz[:,iloc,isrc] / strain[0,:,iloc,isrc], ptrn[isrc], label=f"tilde {src}")
        
        if isrc == 0 and showAllTheSame:
            ax[0].plot(locs, SimpleTilde, ptrn[isrc] , label=f"all the same")
            ax[1].plot(locs, SimpleTilde, ptrn[isrc], label=f"all the same {src}")
            ax[2].plot(locs, SimpleTilde, ptrn[isrc], label=f"all the same {src}")

    # plt.plot(locs, TildeMxL , label="tilde L")
    # plt.plot(locs, TildeMxU , label="tilde U")
    ax[0].set_ylabel('fMx')
    ax[1].set_ylabel('fMy')
    ax[2].set_ylabel('fFz')
    ax[2].set_xlabel("r/R")
    ax[0].set_title("factors" + loc)
    plt.legend()
    plt.savefig(mydir+f"/factors_{iloc}.png")


# # WEIGHTING FACTOR RELATIVE
# # How much weighting we have compared to how much weighting the regular approach would put.
# for iloc,loc in enumerate(leg_loc):
#     fig,ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 5))
    
#     for isrc,src in enumerate(leg_src):
#         #WHEN WE CONSIDER THAT ALL THINGS HAVE THE SAME WEIGHT
#         SimpleTilde = 1 / (yoEIxx[0,:,iloc] + xoEIyy[0,:,iloc] + ooEA[0,:])  #scaling the strain to the proper units

#         ax[0].plot(locs, TildeMx[:,iloc,isrc] / strain[0,:,iloc,isrc] * yoEIxx[0,:,iloc], ptrn[isrc], label=f"tilde {src}")
#         ax[1].plot(locs, TildeMy[:,iloc,isrc] / strain[0,:,iloc,isrc] * xoEIyy[0,:,iloc], ptrn[isrc], label=f"tilde {src}")
#         ax[2].plot(locs, TildeFz[:,iloc,isrc] / strain[0,:,iloc,isrc] * ooEA[0,:], ptrn[isrc], label=f"tilde {src}")
        
#         if isrc == 0 and showAllTheSame:
#             ax[0].plot(locs, SimpleTilde * yoEIxx[0,:,iloc], ptrn[isrc] , label=f"all the same")
#             ax[1].plot(locs, SimpleTilde * xoEIyy[0,:,iloc], ptrn[isrc], label=f"all the same {src}")
#             ax[2].plot(locs, SimpleTilde * ooEA[0,:], ptrn[isrc], label=f"all the same {src}")

#     # plt.plot(locs, TildeMxL , label="tilde L")
#     # plt.plot(locs, TildeMxU , label="tilde U")
#     ax[0].set_ylabel('fMx REL')
#     ax[1].set_ylabel('fMy REL')
#     ax[2].set_ylabel('fFz REL')
#     ax[2].set_xlabel("r/R")
#     ax[0].set_title("factorsREL" + loc)
#     plt.legend()
#     plt.savefig(f"factorsREL_{iloc}.png")



plt.show()

        