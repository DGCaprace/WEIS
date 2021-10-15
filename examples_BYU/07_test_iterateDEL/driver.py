import os
import yaml

from wisdem import run_wisdem
from weis.glue_code.runWEIS import run_weis
from wisdem.inputs import load_yaml, write_yaml #, validate_without_defaults, validate_with_defaults, simple_types

# from pCrunch.io import OpenFASTOutput
from pCrunch import PowerProduction#, LoadsAnalysis

import sys, shutil
import numpy as np
import matplotlib.pyplot as plt


# ---------------------
def my_write_yaml(instance, foutput):
    if not os.path.isfile(foutput):
        print(f"File {foutput} already exists... replacing it.")
        os.remove(foutput)
    # Write yaml with updated values
    with open(foutput, "w", encoding="utf-8") as f:
        yaml.dump(instance, f)


#TODO: make sure we use same turbine between part1 and part2 -> data in yaml and in openfast file should match
# ---> can we regenerate ALL openfast files using only info from the yaml??

#==================== DEFINITIONS  =====================================

## File management
mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
# fname_wt_input = mydir + os.sep + "IEA-10-198-RWT.yaml"
fname_wt_input = mydir + os.sep + "Madsen2019_10_forWEIS.yaml"
fname_modeling_options = mydir + os.sep + "modeling_options.yaml"
fname_analysis_options = mydir + os.sep + "analysis_options_struct.yaml"
fname_analysis_options_WEIS = mydir + os.sep + "analysis_options_WEIS.yaml"
fname_aggregatedEqLoads = mydir + os.sep + "aggregatedEqLoads.yaml"

folder_arch = mydir + os.sep + "results"


#location of servodyn lib (./local of weis)
run_dir1            = "/Users/dg/Documents/BYU/devel/Python/WEIS"
run_dir2            = mydir + os.sep + ".." + os.sep + "Madsen2019_model_BD"

withDEL = True  #skip DEL computation
doLofiOptim = True  #skip lofi optimization
nGlobalIter = 1
restartAt = 0


#==================== ======== =====================================
## Preprocessing

if not withDEL and not doLofiOptim: nGlobalIter = 0
if not withDEL or not doLofiOptim: nGlobalIter = 1

# analysis_opt = load_yaml(fname_analysis_options)
wt_init = load_yaml(fname_wt_input)

#write the WEIS input file
analysis_options_WEIS = {}
analysis_options_WEIS["general"] = {}
analysis_options_WEIS["general"]["folder_output"] = "outputs_WEIS"
analysis_options_WEIS["general"]["fname_output"] = "DTU10MW_Madsen"

my_write_yaml(analysis_options_WEIS, fname_analysis_options_WEIS)

# Restart from a previous iteration:
restartAt = max(0,restartAt)
current_wt_input = fname_wt_input
if restartAt > 0:
    folder_wt_restart = folder_arch + os.sep + "outputs_optim" + os.sep + f"iter_{restartAt-1}"
    if not os.path.isdir(folder_wt_restart):
        raise FileNotFoundError(f"Can't restart from iter {restartAt-1} in folder {folder_wt_restart}")     
    current_wt_input = folder_wt_restart + os.sep + "blade_out.yaml"
        

#==================== ======== =====================================
# Unsteady loading computation from DLCs

for IGLOB in range(restartAt,nGlobalIter):
    print("\n\n\n  ============== ============== ===================\n")
    print(f"  ============== GLOBAL ITER {IGLOB} ===================\n")
    print("  ============== ============== ===================\n\n\n\n")


    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #           PHASE 1 : Compute DEL and extrapolate extreme loads
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if withDEL:
        # Run the base simulation
        wt_opt, modeling_options, opt_options = run_weis(
            current_wt_input, fname_modeling_options, fname_analysis_options_WEIS
        )


        print("\n\n\n  -------------- DONE WITH WEIS ------------------\n\n\n\n")
        sys.stdout.flush()

        # ----------------------------------------------------------------------------------------------
        #    specific preprocessing and definitions

        # nt = len(ct[0]["B1N001FLz"])
        nx = modeling_options["WISDEM"]["RotorSE"]["n_span"]
        nx_hard = 40 #hardcoded in runFAST_pywrapper
        if nx > nx_hard: 
            raise RuntimeError("Not enough channels for DELs provisionned in runFAST_pywrapper.")

        dnx = 1 #if you want to reduce the number of data by step of dnx
        # dnt = 10

        # Design choice: for how long do you size the turbine + other parameters
        m_wohler = 10 #caution: also hardcoded in the definition of fatigue_channels at the top of runFAST_pywrapper
        Tlife = 3600 * 24 * 365 * 20 #the design life of the turbine, in seconds (20 years)
        f_eq = 1 #rotor rotation freq is around 0.1Hz. Let's multiply by 10...100  -- THIS IS TOTALLY ARBITRARY FOR NOW

        fac = np.array([1.,1.,1.e3,1.e3,1.e3]) #multiplicator because output of AD is in N, but output of ED is in kN

        # ----------------------------------------------------------------------------------------------
        #   reading data
        
        # Init our lifetime DEL
        DEL_life_B1 = np.zeros([nx,5])    

        #  -- Retreive the DELstar --
        # (after removing "elapsed" from the del post_processing routine in weis)
        npDelstar = wt_opt['aeroelastic.DELs'].to_numpy()

        fast_fnames = wt_opt['aeroelastic.DELs'].index #the names in here are the filename prefix
        fast_dlclist = wt_opt['aeroelastic.dlc_list']


        #duration of  time series
        Tj = modeling_options["Level3"]["simulation"]["TMax"] - modeling_options["Level3"]["simulation"]["TStart"]

        #number of time series
        Nj = len(npDelstar)
        print(f"Found {Nj} time series...")
        wt_opt['aeroelastic.DELs'].info()

        # Indices where to find DELs for the various nodes:
        colnames = wt_opt['aeroelastic.DELs'].columns
        i_AB1Fn = np.zeros(nx,int)
        i_AB1Ft = np.zeros(nx,int)
        i_B1MLx = np.zeros(nx,int)
        i_B1MLy = np.zeros(nx,int)
        i_B1FLz = np.zeros(nx,int)
        for i in range(nx):
            # i_AB1Fn[i] = colnames.get_loc("AB1N%03iFn"%(i+1)) #local chordwise
            # i_AB1Ft[i] = colnames.get_loc("AB1N%03iFt"%(i+1)) #local normal
            i_AB1Fn[i] = colnames.get_loc("AB1N%03iFx"%(i+1)) #rotor normal
            i_AB1Ft[i] = colnames.get_loc("AB1N%03iFy"%(i+1)) #rotor tangential
            i_B1MLx[i] = colnames.get_loc("B1N%03iMLx"%(i+1))
            i_B1MLy[i] = colnames.get_loc("B1N%03iMLy"%(i+1))
            i_B1FLz[i] = colnames.get_loc("B1N%03iFLz"%(i+1))

        # ----------------------------------------------------------------------------------------------
        # -- Compute extrapolated lifetime DEL for life --

        # Identify what time series correspond to DEL - as per IEC standard, we use NTW -- labeled DLC 1.1
        jDEL = []
        for j in range(Nj):
            if 1.1 == fast_dlclist[j]:
                jDEL.append(j)        
        
        if not jDEL:
            raise Warning("I did not find required data among time series to compute DEL! They will end up being 0.")
        else:
            print(f"Time series {jDEL} are being processed for DEL...")

        #probability of the turbine to operate in specific conditions. For now let's assume uniform distribution. 
        pp = PowerProduction(wt_init['assembly']['turbine_class'])
        iec_dlc_for_del = 1.1 #hardcoded
        iec_dlc_for_extr = 1.3 #hardcoded
        iec_settings = modeling_options["openfast"]["dlc_settings"]["IEC"]
        iec_del = {}
        iec_extr = {}
        for dlc in iec_settings:
            if iec_dlc_for_del == dlc["DLC"]:
                iec_del = dlc
                Udel_str = iec_del["U"]
            if iec_dlc_for_extr == dlc["DLC"]:
                iec_extr = dlc

        pj = pp.prob_WindDist([float(u) for u in Udel_str], disttype='pdf')
        pj = pj / np.sum(pj) #renormalizing so that the sum of all the velocity we simulated covers the entire life of the turbine
        #--
        # pj = np.ones(Nj) / Nj   #uniform probability instead

        print("Weight of the series (probability):")
        print(pj)
        

        # a. Obtain the equivalent number of cycles
        fj = Tlife / Tj * pj
        n_life_eq = np.sum(fj * Tj * f_eq)
        
        # b. Aggregate DEL
        k=0
        for ids in [i_AB1Fn,i_AB1Ft,i_B1MLx,i_B1MLy,i_B1FLz]:
            #loop over the DELs from all time series and sum
            for j in jDEL:
                DEL_life_B1[:,k] += fj[j] * npDelstar[j][ids] 
            k+=1
        DEL_life_B1 = .5 * fac * ( DEL_life_B1 / n_life_eq ) ** (1/m_wohler)

        # More processing:
        #1) switch from IEC local blade frame to "airfoil frame" with x towards TE
        tmp = DEL_life_B1[:,2].copy()
        DEL_life_B1[:,2] = -DEL_life_B1[:,3]
        DEL_life_B1[:,3] = tmp
        DEL_life_B1[:,4] = -DEL_life_B1[:,4] #change sign because RotorSE strain computation considers positive loads are compression??

        print("Damage eq loads:")
        print(np.transpose(DEL_life_B1))


        # ----------------------------------------------------------------------------------------------
        # -- proceed to extreme load/gust extrapolation if requested

        # Identify what time series correspond to extreme loads - as per IEC standard, we use ETW -- labeled DLC 1.3
        jEXTR = []
        for j in range(Nj):
            if 1.3 == fast_dlclist[j]:
                jEXTR.append(j)  
        
        if not jEXTR:
            raise Warning("I did not find required data among time series to compute extreme loads! They will end up being 0.")
        else:
            print(f"Time series {jEXTR} are being processed for extreme loads...")


        # ----------------------------------------------------------------------------------------------
        # -- Create a descriptor:

        #Just a string written in the DEL export files to describe what's in there
        if iec_del:
            del_descr_str = "DEL computed based on DLC %f with %i seeds and the following vels: %s"%(
                iec_del['DLC'], len(iec_del['Seeds']), iec_del['U']
            )
        else:
            del_descr_str = "DEL unavailable"

        if iec_extr:
            extr_descr_str = "extreme loading computed based on DLC %f with %i seeds and the following vels: %s"%(
                iec_extr['DLC'], len(iec_extr['Seeds']), iec_extr['U']
            )
        else:
            extr_descr_str = "extreme loading unavailable"

        # ----------------------------------------------------------------------------------------------
        # -- write the analysis file

        schema = load_yaml(fname_analysis_options)
        #could use load_analysis_yaml from weis instead

        schema["DEL"] = {}
        schema["DEL"]["description"] = del_descr_str
        schema["DEL"]["grid_nd"] = np.linspace(0.,1.,nx).tolist() #note: the node gauges are located at np.arange(1./nx/2., 1, 1./nx) but I prefer consider that it spans then entire interval [0,1]
        schema["DEL"]["deMLx"] = DEL_life_B1[:,2].tolist()
        schema["DEL"]["deMLy"] = DEL_life_B1[:,3].tolist()
        schema["DEL"]["deFLz"] = DEL_life_B1[:,4].tolist()
        # schema["extreme"] = {}

        schema["general"]["folder_output"] = "outputs_struct_withFatigue"
        schema["constraints"]["blade"]["fatigue_spar_cap_ss"]["flag"] = True
        schema["constraints"]["blade"]["fatigue_spar_cap_ps"]["flag"] = True
        schema["constraints"]["blade"]["fatigue_spar_cap_ss"]["eq_Ncycle"] = float(n_life_eq)
        schema["constraints"]["blade"]["fatigue_spar_cap_ps"]["eq_Ncycle"] = float(n_life_eq)
        schema["constraints"]["blade"]["fatigue_spar_cap_ss"]["m_wohler"] = m_wohler
        schema["constraints"]["blade"]["fatigue_spar_cap_ps"]["m_wohler"] = m_wohler

        fname_analysis_options_struct = mydir + os.sep + "analysis_options_struct_withDEL.yaml"
        my_write_yaml(schema, fname_analysis_options_struct)
        #could use write_analysis_yaml from weis instead
        #TODO: save in a format that can be used by MACH

        schema_hifi = {}
        schema_hifi["DEL"] = {}
        schema_hifi["DEL"]["description"] = del_descr_str
        schema_hifi["DEL"]["grid_nd"] = schema["DEL"]["grid_nd"]
        schema_hifi["DEL"]["Fn"] = DEL_life_B1[:,0].tolist()
        schema_hifi["DEL"]["Ft"] = DEL_life_B1[:,1].tolist()
        # schema_hifi["extreme"] = {}

        my_write_yaml(schema_hifi, fname_aggregatedEqLoads)

    else:
        fname_analysis_options_struct = mydir + os.sep + "analysis_options_struct.yaml"


    # +++++++++++++++++++++++++++++++++++++++
    #           PHASE 2 : Optimize
    # +++++++++++++++++++++++++++++++++++++++
    if doLofiOptim:
        # Let's use the most up-to-date turbine as a starting point:
        wt_opt, analysis_options, opt_options = run_wisdem(current_wt_input, fname_modeling_options, fname_analysis_options_struct)

        print("\n\n\n  -------------- DONE WITH WISDEM ------------------\n\n\n\n")    

    # +++++++++++++++++++++++++++++++++++++++
    #           PHASE 3 : book keeping
    # +++++++++++++++++++++++++++++++++++++++

    if not os.path.isdir(folder_arch):
        os.makedirs(folder_arch)

    currFolder = f"iter_{IGLOB}"

    
    # shutil.copy(os.path.join(fileDirectory,file), os.path.join(workingDirectory,file))
    # shutil.copytree
    if withDEL and IGLOB==0:
        shutil.move(fname_aggregatedEqLoads,folder_arch+os.sep)
    if os.path.isdir(mydir + os.sep + "outputs_WEIS"):
        shutil.move(mydir + os.sep + "outputs_WEIS", folder_arch+ os.sep + "outputs_WEIS" + os.sep + currFolder)  
        shutil.move(mydir + os.sep + "temp", folder_arch + os.sep + "sim" + os.sep + currFolder)
    if os.path.isdir(mydir + os.sep + "outputs_struct_withFatigue"):
        shutil.move(mydir + os.sep + "outputs_struct_withFatigue", folder_arch + os.sep + "outputs_optim" + os.sep + currFolder)
    if os.path.isdir(mydir + os.sep + "outputs_struct"):
        shutil.move(mydir + os.sep + "outputs_struct", folder_arch + os.sep + "outputs_optim" + os.sep + currFolder)

    # update the path to the current optimal turbine
    current_wt_input = folder_arch + os.sep + "outputs_optim" + os.sep + currFolder + os.sep + "blade_out.yaml"


## -- plot successive DEL --

print(f"  ============== DONE AFTER {nGlobalIter} ITER ===================\n")