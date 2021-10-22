import numpy as np
import matplotlib.pyplot as plt
from wisdem.inputs import load_yaml


WISDEMout = "results-IEC1.1_5vels_120s_4Glob/outputs_optim/iter_0/blade_out.npz"
#--
EXTRAPout = "results-IEC1.1-IEC1.3_5vels_120s_0Glob_chi2/"
# EXTRAPout = "results-IEC1.1-IEC1.3_5vels_120s_0Glob_norm/"


# WISDEMout = "optim_IEC1.3_5vels_normEXTR/outputs_optim/iter_0/blade_out.npz"
# WISDEMout = "optim_IEC1.3_5vels_normEXTR_noMinus/outputs_optim/iter_0/blade_out.npz"
WISDEMout = "optim_IEC1.3_5vels_normEXTR_noMinus_noRot/outputs_optim/iter_0/blade_out.npz"
WISDEMout2 = "optim_IEC1.3_5vels_normEXTR_noRot/outputs_optim/iter_0/blade_out.npz"


mya = {}
myb = {}
myc = {}

# with np.load(WISDEMout) as a:
#     # print(a.files)

#     mya['r']  = np.array(a["rotorse.rs.z_az_m"])
#     mya['r']  = (mya['r']-mya['r'][0])/(mya['r'][-1]-mya['r'][0])
#     mya['M1'] = np.array(a["rotorse.rs.strains.M1_N*m"])
#     mya['M2'] = np.array(a["rotorse.rs.strains.M2_N*m"])
#     mya['F3'] = np.array(a["rotorse.rs.strains.F3_N"])
#     mya['Fn'] = np.array(a["rotorse.rs.tot_loads_gust.aeroloads_Px_N/m"])
#     mya['Ft'] = -np.array(a["rotorse.rs.tot_loads_gust.aeroloads_Py_N/m"])  # WHY WOULD WE NEED TO PUT A NEGATIVE HERE??

#     mya['sU'] = a["rotorse.rs.strains.strainU_spar"]
#     mya['sL'] = a["rotorse.rs.strains.strainL_spar"]
#     myb['sU'] = a["rotorse.rs.extreme_strains.strainU_spar"]
#     myb['sL'] = a["rotorse.rs.extreme_strains.strainL_spar"]

# b = load_yaml(EXTRAPout + "aggregatedEqLoads.yaml")
# c = load_yaml(EXTRAPout + "analysis_options_struct_withDEL.yaml")

# myb['r']  = np.array(c["extreme"]["grid_nd"])
# myb['M1'] = np.array(c["extreme"]["deMLx"]) #after switching axes
# myb['M2'] = np.array(c["extreme"]["deMLy"]) #after switching axes
# myb['F3'] = -np.array(c["extreme"]["deFLz"]) #NEED TO PUT A - SIGN HERE, CAUSE THERE IS STILL A BUG IN HANSEN'S FORMULA!!!
# myb['Fn'] = np.array(b["extreme"]["Fn"])
# myb['Ft'] = np.array(b["extreme"]["Ft"]) 

# x = ["r","r","r","r","r"]
# toPlot = ["M1","M2", "F3", "Fn", "Ft"]
# fac = [1, 1, 1, 1, 1]

with np.load(WISDEMout) as a:
    # print(a.files)

    mya['r']  = np.array(a["rotorse.rs.z_az_m"])
    mya['r']  = (mya['r']-mya['r'][0])/(mya['r'][-1]-mya['r'][0])
    mya['M1'] = a["rotorse.rs.strains.M1_N*m"]
    mya['M2'] = a["rotorse.rs.strains.M2_N*m"]
    mya['F3'] = a["rotorse.rs.strains.F3_N"]

    mya['sU'] = a["rotorse.rs.strains.strainU_spar"]
    mya['sL'] = a["rotorse.rs.strains.strainL_spar"]
    mya['cU'] = a["rotorse.rs.constr.constr_max_strainU_spar"]
    mya['cL'] = a["rotorse.rs.constr.constr_max_strainL_spar"]
    mya['four'] = range(4)


    myb['r']  = mya['r']
    myb['M1'] = a["rotorse.rs.extreme_strains.M1_N*m"]
    myb['M2'] = a["rotorse.rs.extreme_strains.M2_N*m"]
    myb['F3'] = a["rotorse.rs.extreme_strains.F3_N"]

    myb['sU'] = a["rotorse.rs.extreme_strains.strainU_spar"]
    myb['sL'] = a["rotorse.rs.extreme_strains.strainL_spar"]
    myb['cU'] = a["rotorse.rs.constr.constr_extreme_strainU_spar"]
    myb['cL'] = a["rotorse.rs.constr.constr_extreme_strainL_spar"]
    myb['four'] = range(4)


with np.load(WISDEMout2) as a:
    myc['r']  = mya['r']
    myc['M1'] = a["rotorse.rs.extreme_strains.M1_N*m"]
    myc['M2'] = a["rotorse.rs.extreme_strains.M2_N*m"]
    myc['F3'] = a["rotorse.rs.extreme_strains.F3_N"]

    myc['sU'] = a["rotorse.rs.extreme_strains.strainU_spar"]
    myc['sL'] = a["rotorse.rs.extreme_strains.strainL_spar"]
    myc['cU'] = a["rotorse.rs.constr.constr_extreme_strainU_spar"]
    myc['cL'] = a["rotorse.rs.constr.constr_extreme_strainL_spar"]
    myc['four'] = range(4)

toPlot = ["M1","M2", "F3",'sU', "sL"] #, 'cU', "cL"
x = ["r","r","r","r","r","four","four"]
fac = [1, 1, 1, 1, 1, 1, 1]

print(myc['cL'])

# -----------------------------------------------------------
leg = toPlot


pmax = len(toPlot)
xmax = 1.#len(histValues[toPlot[0]])


fig, ax = plt.subplots(nrows=pmax, ncols=1, figsize=(10, 1.2*pmax))
for i,p,f,l in zip(range(pmax),toPlot,fac,leg):
    ax[i].plot(mya[x[i]],mya[p]/f,'--')
    ax[i].plot(myb[x[i]],myb[p]/f,'-')
    ax[i].plot(myc[x[i]],myc[p]/f,'-')
    ax[i].set_ylabel(l)
    ax[i].set_xlim(( myb[x[i]][0],myb[x[i]][-1]))
    # ax[0].set_ylim((1,7.0))

ax[pmax-1].set_xlabel("iterations")

plt.tight_layout()
# plt.savefig("hist.png")
plt.show()
