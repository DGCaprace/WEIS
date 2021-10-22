import numpy as np
import matplotlib.pyplot as plt
from wisdem.inputs import load_yaml


WISDEMout = "results-IEC1.1_5vels_120s_4Glob/outputs_optim/iter_0/blade_out.npz"
#--
EXTRAPout = "results-IEC1.1-IEC1.3_5vels_120s_0Glob_chi2/"
# EXTRAPout = "results-IEC1.1-IEC1.3_5vels_120s_0Glob_norm/"


mya = {}
myb = {}

with np.load(WISDEMout) as a:
    # print(a.files)

    mya['r']  = np.array(a["rotorse.rs.z_az_m"])
    mya['M1'] = np.array(a["rotorse.rs.strains.M1_N*m"])
    mya['M2'] = np.array(a["rotorse.rs.strains.M2_N*m"])
    mya['F3'] = np.array(a["rotorse.rs.strains.F3_N"])
    mya['Fn'] = np.array(a["rotorse.rs.tot_loads_gust.aeroloads_Px_N/m"])
    mya['Ft'] = -np.array(a["rotorse.rs.tot_loads_gust.aeroloads_Py_N/m"])  # WHY WOULD WE NEED TO PUT A NEGATIVE HERE??


b = load_yaml(EXTRAPout + "aggregatedEqLoads.yaml")
c = load_yaml(EXTRAPout + "analysis_options_struct_withDEL.yaml")

myb['r']  = np.array(b["extreme"]["grid_nd"])
myb['M1'] = np.array(c["extreme"]["deMLx"]) #after switching axes
myb['M2'] = np.array(c["extreme"]["deMLy"]) #after switching axes
myb['F3'] = -np.array(c["extreme"]["deFLz"]) #NEED TO PUT A - SIGN HERE, CAUSE THERE IS STILL A BUG IN HANSEN'S FORMULA!!!
myb['Fn'] = np.array(b["extreme"]["Fn"])
myb['Ft'] = np.array(b["extreme"]["Ft"]) 



toPlot = ["M1","M2", "F3", "Fn", "Ft"]
fac = [1, 1, 1, 1, 1]
leg = toPlot


pmax = len(toPlot)
xmax = 1.#len(histValues[toPlot[0]])


fig, ax = plt.subplots(nrows=pmax, ncols=1, figsize=(10, 1.2*pmax))
for i,p,f,l in zip(range(pmax),toPlot,fac,leg):
    ax[i].plot((mya['r']-mya['r'][0])/(mya['r'][-1]-mya['r'][0]),mya[p]/f,'-')
    ax[i].plot(myb['r'],myb[p]/f,'-')
    ax[i].set_ylabel(l)
    ax[i].set_xlim((0,xmax))
    # ax[0].set_ylim((1,7.0))

ax[pmax-1].set_xlabel("iterations")

# plt.savefig("hist.png")

plt.show()
