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

def my_write_yaml(instance, foutput):
    if os.path.isfile(foutput):
        print(f"File {foutput} already exists... replacing it.")
        os.remove(foutput)
    # Write yaml with updated values
    with open(foutput, "w", encoding="utf-8") as f:
        yaml.dump(instance, f)

def my_read_yaml(finput):
    # Write yaml with updated values
    with open(finput, "r", encoding="utf-8") as f:
        dict = yaml.load(f, Loader=yaml.FullLoader)
    return dict


#==================== DEFINITIONS  =====================================

## File management
mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file

wt_input = [  "Madsen2019_10_forWEIS.yaml",
                    # "Madsen2019_10_forWEIS_isotropic.yaml",
                    "Madsen2019_10_forWEIS_isotropic_ED.yaml",
                    # "Madsen2019_10_forWEIS_isotropic_BD.yaml", #there is a bug somewhere there
                    ]
# wt_input = ["Madsen2019_10_forWEIS_isotropic.yaml"]
# wt_input = ["Madsen2019_10_forWEIS.yaml"]
# wt_input = ["Madsen2019_10_forWEIS_isotropic_IC_NEW.yaml"]

wt_input = ["Madsen2019_10_forWEIS_isotropic_IC.yaml"]


fname_modeling_options = mydir + os.sep + "modeling_options.yaml"
fname_analysis_options_WEIS = mydir + os.sep + "analysis_options_WEIS.yaml"


# Need to define these parameters since we do not use explicitely the drivetrain model, and BeamDyn requires them when GenDOF is on.
HubMass  = 105.52E3   
HubIner  = 325.6709E3 
GenIner  = 1500.5     
NacMass  = 446.03625E3
NacYIner = 7326.3465E3

plotOnly = False
fast_fnames = ["weis_job_0"]

#==================== RUN THE TURBINE WITH WEIS, FROM YAML INPUTS =====================================
# Loading the wisdem/weis compatible yaml, and propagate information to aeroelasticse

# Construct a dict with values to overwrite
overridden_values = {}
overridden_values["aeroelastic.hub_system_mass"] = [HubMass,]
overridden_values["aeroelastic.hub_system_I"]    = [HubIner,0,0,0,0,0]
overridden_values["aeroelastic.GenIner"]         = GenIner
overridden_values["aeroelastic.above_yaw_mass"]  = [NacMass,]
overridden_values["aeroelastic.nacelle_I_TT"]    = [0,0,NacYIner,0,0,0]


flds = ['mean','std','min','max']

nx = 9 #XXX HARDCODED: number of "stations"
nchan = 3 #XXX HARDCODED: displacement in 3 dirs
data = np.zeros((nchan,nx,len(flds),len(wt_input)))
chans_leg = ['Dx','Dy','Dz']

nx_a = 30 #XXX HARDCODED: number of "nodes"
nchan_a = 7 #XXX HARDCODED: Fn, Ft, Mx, My, Fz, SparStrainU, SparStrainL
data_a = np.zeros((nchan_a,nx_a,len(flds),len(wt_input)))
chans_a_leg = ["Fn"," Ft"," Mx"," My"," Fz","SparStrainU","SparStrainL"]


locs = np.linspace(0.,1.,nx_a) #XXX




for ifi in range(len(wt_input)):
    wt_file = wt_input[ifi]

    folder_arch = mydir + os.sep + wt_file.split(".")[0]
    fname_wt_input = mydir + os.sep + wt_file

    fname_nominalLoads = folder_arch + os.sep + "nominalLoads.yaml"

    if not os.path.isdir(folder_arch):
        os.makedirs(folder_arch)


    if not plotOnly:

        # Run the modified simulation with the overwritten values
        wt_opt, modeling_options, opt_options = run_weis(
            fname_wt_input, fname_modeling_options, fname_analysis_options_WEIS,
            overridden_values=overridden_values,
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


        summary_stats = wt_opt['aeroelastic.summary_stats']

        print("\n\n\n  -------------- DONE WITH WEIS ------------------\n")
        print("\n\n\n")

        simfolder = mydir + os.sep + modeling_options["openfast"]["file_management"]["FAST_runDirectory"]

        #moving stuff around
        if os.path.isdir(mydir + os.sep + "outputs_WEIS"):
            shutil.move(mydir + os.sep + "outputs_WEIS", folder_arch+ os.sep + "outputs_WEIS")  
        if os.path.isdir(simfolder): #let's not move the file if it is a path provided by the user
            shutil.move(simfolder, folder_arch + os.sep + "sim")
        os.system(f"cp {fname_wt_input} {folder_arch}")
        os.system(f"cp {fname_modeling_options} {folder_arch}")


        #==================== POST PROCESSING =====================================

        if modeling_options["Level3"]["simulation"]["CompElast"] == 1: #Elastodyn
            chans = ["Spn%1iTDxb1","Spn%1iTDyb1","Spn%1iRDzb1"] #displacements
        else: #BeamDyn
            chans = ["B1N%1iTDxr","B1N%1iTDyr","B1N%1iRDzr"] #RDzr  is Wiener-Malenkoviz
        fact = [1.,1.,1.] #displacements are all in m
            
        chans_a = ["AB1N%03iFx","AB1N%03iFy"]
        fact_a = [1.,1.] #aero forces in N
        if modeling_options["Level3"]["simulation"]["CompElast"] == 1: #Elastodyn
            chans_a.extend(["B1N%03iMLx","B1N%03iMLy","B1N%03iFLz"]) #internal/residual moments and forces
            fact_a.extend([1.e3,1.e3,1.e3]) #ED beam stuff in kN
        else: #BeamDyn
            chans_a.extend(["B1N%03i_MxL","B1N%03i_MyL","B1N%03i_Fzr"]) #internal/residual moments and forces
            fact_a.extend([1.,1.,1.]) #BD beam stuff in N
        
        #strain
        chans_a.extend(['BladeSparU_Strain_Stn%i','BladeSparL_Strain_Stn%i'])
        fact_a.extend([1.,1.])


        #data spanning over the stations
        k = 0
        for lab in chans:
            for i in range(nx):
                for ifld,fld in enumerate(flds):
                    data[k,i,ifld,ifi] = summary_stats[lab%(i+1)][fld] * fact[k]
            k+=1

        #data spanning over the nodes
        k = 0
        for lab in chans_a:
            for i in range(nx_a):
                for ifld,fld in enumerate(flds):
                    data_a[k,i,ifld,ifi] = summary_stats[lab%(i+1)][fld] * fact_a[k]
            k+=1


        #==================== Export the nominal loads ====================
        
        run_settings = modeling_options["DLC_driver"]["DLCs"][0] 

        schema = {}

        schema["description"] = f"nominal loads obtained for inflow velocity {run_settings['wind_speed']}"
        schema["grid_nd"] = locs.tolist()

        schema["nominal"] = {}
        for ifld,fld in enumerate(flds):
            schema["nominal"][fld] = {}
            schema["nominal"][fld]["Dx"] = data[0,:,ifld,ifi].tolist()
            schema["nominal"][fld]["Dy"] = data[1,:,ifld,ifi].tolist()
            schema["nominal"][fld]["Dz"] = data[1,:,ifld,ifi].tolist()
            schema["nominal"][fld]["Fn"] = data_a[0,:,ifld,ifi].tolist()
            schema["nominal"][fld]["Ft"] = data_a[1,:,ifld,ifi].tolist()
            schema["nominal"][fld]["deMLx"] = data_a[2,:,ifld,ifi].tolist()
            schema["nominal"][fld]["deMLy"] = data_a[3,:,ifld,ifi].tolist() #positive towards TE
            schema["nominal"][fld]["deFLz"] = data_a[4,:,ifld,ifi].tolist()
            schema["nominal"][fld]["SparStrainU"] = data_a[5,:,ifld,ifi].tolist()
            schema["nominal"][fld]["SparStrainL"] = data_a[6,:,ifld,ifi].tolist()

        my_write_yaml(schema, fname_nominalLoads)


    else: 
        curr_modeling_options = fname_modeling_options.split(os.sep)[-1]
        if os.path.isdir(curr_modeling_options):
            modeling_options = load_yaml(curr_modeling_options) 
        else:
            modeling_options = load_yaml(fname_modeling_options) 

        schema = my_read_yaml(fname_nominalLoads)

        for ifld,fld in enumerate(flds):
            data[0,:,ifld,ifi] = schema["nominal"][fld]["Dx"]
            data[1,:,ifld,ifi] = schema["nominal"][fld]["Dy"]
            data[1,:,ifld,ifi] = schema["nominal"][fld]["Dz"]
            data_a[0,:,ifld,ifi] = schema["nominal"][fld]["Fn"]
            data_a[1,:,ifld,ifi] = schema["nominal"][fld]["Ft"]
            data_a[2,:,ifld,ifi] = schema["nominal"][fld]["deMLx"]
            data_a[3,:,ifld,ifi] = schema["nominal"][fld]["deMLy"]
            data_a[4,:,ifld,ifi] = schema["nominal"][fld]["deFLz"]
            data_a[5,:,ifld,ifi] = schema["nominal"][fld]["SparStrainU"]
            data_a[6,:,ifld,ifi] = schema["nominal"][fld]["SparStrainL"]
  


#==================== Compare  Plots #====================

#deflections
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(15, 5))

for ifi in range(len(wt_input)):
    wt_file = wt_input[ifi]

    for k in range(nchan):
        hp = ax[k].plot(data[k,:,0,ifi],'x-', label=wt_file)
        
        ax[k].plot(data[k,:,0,ifi]+data[k,:,1,ifi],'--', color=hp[0].get_color())
        ax[k].plot(data[k,:,0,ifi]-data[k,:,1,ifi],'--', color=hp[0].get_color())
    
        ax[k].set_ylabel(f"{chans_leg[k]}")
# plt.xlabel("U [m/s]")

plt.legend()

fig.savefig(folder_arch + "/deflections.png")



#loads
fig, axa = plt.subplots(nrows=nchan_a, ncols=1, figsize=(15, 1.5*nchan_a))

for ifi in range(len(wt_input)):
    wt_file = wt_input[ifi]

    for k in range(nchan_a):
        hp = axa[k].plot(locs,data_a[k,:,0,ifi],'x-', label=wt_file)
        
        axa[k].plot(locs,data_a[k,:,0,ifi]+data_a[k,:,1,ifi],'--', color=hp[0].get_color())
        axa[k].plot(locs,data_a[k,:,0,ifi]-data_a[k,:,1,ifi],'--', color=hp[0].get_color())
    
        axa[k].set_ylabel(f"{chans_a_leg[k]}")
    
        axa[k].autoscale_view(tight=True)
plt.tight_layout()

fig.savefig(folder_arch + "/nominal_loads.png")

plt.show()