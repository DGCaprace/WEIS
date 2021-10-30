import os
import yaml

from wisdem import run_wisdem
from weis.glue_code.runWEIS import run_weis
from wisdem.inputs import load_yaml, write_yaml #, validate_without_defaults, validate_with_defaults, simple_types
from pCrunch import PowerProduction, LoadsAnalysis
from pCrunch.io import OpenFASTAscii, OpenFASTBinary#, OpenFASTOutput

import sys, shutil
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def myOpenFASTread(fname,addExt=0):
    if addExt>0 and not ".out" in fname:
        ext = ".outb"
        if  addExt == 1: ext = ".out"
        fname += ext #appending the right extension
    try:
        output = OpenFASTAscii(fname) #magnitude_channels=self._mc
        output.read()

    except UnicodeDecodeError:
        output = OpenFASTBinary(fname) #magnitude_channels=self._mc
        output.read()

    return output

#==================== DEFINITIONS  =====================================

## File management
mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file

wt_input = [  "Madsen2019_10_forWEIS.yaml",
                    "Madsen2019_10_forWEIS_isotropic.yaml"]
# wt_input = ["Madsen2019_10_forWEIS_isotropic.yaml"]
# wt_input = ["Madsen2019_10_forWEIS.yaml"]

fname_modeling_options = mydir + os.sep + "modeling_options.yaml"
# fname_analysis_options = mydir + os.sep + "analysis_options_struct.yaml"
fname_analysis_options_WEIS = mydir + os.sep + "analysis_options_WEIS.yaml"

# I look for the steady solution. 

plotOnly = True
fast_fnames = ["DTU10MW_powercurve_0"]

#==================== RUN THE TURBINE WITH WEIS, FROM YAML INPUTS =====================================
# Loading the wisdem/weis compatible yaml, and propagate information to aeroelasticse


nx = 9
chans = ["Spn%1iTDxb1","Spn%1iTDyb1","Spn%1iRDzb1"]
data_avg = np.zeros((len(chans),nx,len(wt_input)))

for ifi in range(len(wt_input)):
    wt_file = wt_input[ifi]

    folder_arch = mydir + os.sep + wt_file.split(".")[0]
    fname_wt_input = mydir + os.sep + wt_file

    if not os.path.isdir(folder_arch):
        os.makedirs(folder_arch)


    if not plotOnly:
        # Run the base simulation
        wt_opt, modeling_options, opt_options = run_weis(
            fname_wt_input, fname_modeling_options, fname_analysis_options_WEIS
        )

        # ref_V = wt_opt['rotorse.rp.powercurve.V']
        # ref_RPM = wt_opt['rotorse.rp.powercurve.Omega']
        # ref_pitch = np.array(wt_opt['rotorse.rp.powercurve.pitch']) #caution: negative values even though 

        # RPM_weis = wt_opt['aeroelastic.Omega_out']
        # Cp_weis  = wt_opt['aeroelastic.Cp_out']  #aero Cp
        # P_weis  = wt_opt['aeroelastic.P_out']
        # Pitch_weis = wt_opt['aeroelastic.pitch_out']

        # summary_stats = wt_opt['aeroelastic.summary_stats']
        # fast_fnames = summary_stats.index #the names in here are the filename prefix


        print("\n\n\n  -------------- DONE WITH WEIS ------------------\n")
        print("\n\n\n")

        simfolder = mydir + os.sep + modeling_options["openfast"]["file_management"]["FAST_runDirectory"]

        #moving stuff around
        if os.path.isdir(mydir + os.sep + "outputs_WEIS"):
            shutil.move(mydir + os.sep + "outputs_WEIS", folder_arch+ os.sep + "outputs_WEIS")  
        if os.path.isdir(simfolder): #let's not move the file if it is a path provided by the user
            shutil.move(simfolder, folder_arch + os.sep + "sim")
        os.system(f"cp {fname_wt_input} {folder_arch}")


    else: 
        modeling_options = load_yaml(fname_modeling_options)


        # simfolder = mydir + os.sep + modeling_options["openfast"]["file_management"]["FAST_runDirectory"]
        # #moving stuff around
        # if os.path.isdir(mydir + os.sep + "outputs_WEIS"):
        #     shutil.move(mydir + os.sep + "outputs_WEIS", folder_arch+ os.sep + "outputs_WEIS")  
        # if os.path.isdir(simfolder): #let's not move the file if it is a path provided by the user
        #     shutil.move(simfolder, folder_arch + os.sep + "sim")

    
    



    #==================== POST PROCESSING =====================================

    #LOOP OVER THE FILES?

    # Because raw data are not available as such, need to go back look into the output files:
    fname = folder_arch + os.sep + "sim" + os.sep + fast_fnames[0]
    fulldata = myOpenFASTread(fname, addExt=modeling_options["Level3"]["simulation"]["OutFileFmt"])

    k = 0
    for lab in chans:
        for i in range(nx):
            # EXTR_distro_B1[i,k,:] = EXTR_distro_B1[i,k,:] + hist * pj_extr[jloc]
            data_avg[k,i,ifi] = np.mean(fulldata[lab%(i+1)])

        k+=1

    del(fulldata)



## Compare ##
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(15, 5))

for ifi in range(len(wt_input)):
    wt_file = wt_input[ifi]

    for k in range(len(chans)):
        ax[k].plot(data_avg[k,:,ifi],'x-', label=wt_file)
    
        ax[k].set_ylabel(f"{chans[k]%(0)}")
# plt.xlabel("U [m/s]")
    
plt.legend()
plt.show()