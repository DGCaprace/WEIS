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


# ---------------------
def my_write_yaml(instance, foutput):
    if os.path.isfile(foutput):
        print(f"File {foutput} already exists... replacing it.")
        os.remove(foutput)
    # Write yaml with updated values
    with open(foutput, "w", encoding="utf-8") as f:
        yaml.dump(instance, f)


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

def extrapolate_extremeLoads_hist(rng,mat,extr_prob):
    nbins = np.shape(mat)[2]
    n1 = np.shape(mat)[0]
    n2 = np.shape(mat)[1]

    p = np.nan*np.zeros((n1,n2,2))

    for k in range(n2):
        stp = (rng[k][1]-rng[k][0])/(nbins)
        x = np.arange(rng[k][0]+stp/2.,rng[k][1],stp)
        for i in range(n1):
            p[i,k,0] = np.sum( mat[i,k,:] * x ) / np.sum(mat[i,k,:])
            p[i,k,1] = np.sqrt( np.sum( mat[i,k,:] * x**2 ) / np.sum(mat[i,k,:]) - avg[i,k] )
   
    n_std_extreme = stats.norm.ppf(extr_prob)
    EXTR_life_B1 = p[:,:,0] + n_std_extreme * p[:,:,1]

    return EXTR_life_B1, p
    

def extrapolate_extremeLoads(mat, distr_list, extr_prob):
    n1 = np.shape(mat)[0]
    n2 = np.shape(mat)[1]

    extr = np.zeros((n1,n2))

    p = np.nan*np.zeros((n1,n2,3)) #not very general ...

    for k in range(n2):
        distr = getattr(stats,distr_list[k])
        if 'norm' in distr_list[k]:
            for i in range(n1):
                params = distr.fit(mat[i,k,:])
                extr[i,k] = distr.ppf(extr_prob, loc = params[0], scale = params[1])
                # myppf = distr.ppf(extr_prob)
                # extr[i,k] = params[0] + myppf * params[1]
                p[i,k,0] = params[0] #can I do something better than this?
                p[i,k,1] = params[1] #can I do something better than this?
        else:
            #not sure the implementation with loc and scale is super general
            for i in range(n1):
                params = distr.fit(mat[i,k,:])
                extr[i,k] = distr.ppf(extr_prob, params[0], loc=params[1], scale=params[2])
                p[i,k,0] = params[0] #can I do something better than this?
                p[i,k,1] = params[1] #can I do something better than this?
                p[i,k,2] = params[2] #can I do something better than this?
            
    return extr, p


def extrapolate_extremeLoads_curveFit(rng,mat,distr_list, extr_prob):
    nbins = np.shape(mat)[2]
    n1 = np.shape(mat)[0]
    n2 = np.shape(mat)[1]

    thr = 1e-5 #threshold (normalized frequency)

    extr = np.zeros((n1,n2))

    p = np.nan*np.zeros((n1,n2,3)) #not very general ...

    for k in range(n2):
        stp = (rng[k][1]-rng[k][0])/(nbins)
        x = np.arange(rng[k][0]+stp/2.,rng[k][1],stp)
        
        if 'twiceMaxForced'  in distr_list[k]:
            for i in range(n1):
                imax = np.where(mat[i,k,:] >= thr)
                extr[i,k] = 2.*x[imax[0][-1]] if len(imax[0])>0 else 0.0
                
        elif 'normForced' in distr_list[k]:
            for i in range(n1):
                #Curve fitting is a bit sensitive... we could also simply use the good old way.
                # However, it curvefit does not succeed, maybe it is because the distro does not look like a normal at all... 
                #   and would be a good idea not to force that and use a fallback condition instead.
                avg = np.sum( mat[i,k,:] * x ) / np.sum(mat[i,k,:])
                std = np.sqrt( np.sum( mat[i,k,:] * x**2 ) / np.sum(mat[i,k,:]) - avg )
                params = (avg,std)

                extr[i,k] = stats.norm.ppf(extr_prob, loc = params[0], scale = params[1])
                p[i,k,0] = params[0] #can I do something better than this?
                p[i,k,1] = params[1] #can I do something better than this?
        elif 'norm' in distr_list[k]:
            for i in range(n1):
                failed = False
                avg = np.sum( mat[i,k,:] * x ) / np.sum(mat[i,k,:])
                std = np.sqrt( np.sum( mat[i,k,:] * x**2 ) / np.sum(mat[i,k,:]) - avg )
                try: 
                    params, covf = curve_fit(stats.norm.pdf, x, mat[i,k,:], p0 = [avg,std])  #best possible starting point
                    perr = np.sqrt(np.diag(covf))
                    if any(np.isinf(params)) or any(np.isnan(params)) or np.isinf(covf[0][0]) or any(np.isnan(perr)):
                        failed = True
                    extr[i,k] = stats.norm.ppf(extr_prob, loc = params[0], scale = params[1])
                except RuntimeError:   
                    failed = True

                if failed:
                    print(f"Could not determine params for a {distr_list[k]} at {k},{i}. Will just double the max load.")
                    imax = np.where(mat[i,k,:] >= thr)
                    extr[i,k] = 2.*x[imax[0][-1]] if len(imax[0])>0 else 0.0
                    params = (np.nan,0,1)
                
                p[i,k,0] = params[0] #can I do something better than this?
                p[i,k,1] = params[1] #can I do something better than this?
        else:
            distr = getattr(stats,distr_list[k])
            #not sure the implementation with loc and scale is super general though
            # works with chi2
            for i in range(n1):
                failed = False
                avg = np.sum( mat[i,k,:] * x ) / np.sum(mat[i,k,:])
                std = np.sqrt( np.sum( mat[i,k,:] * x**2 ) / np.sum(mat[i,k,:]) - avg )
                try:
                    # params, covf = curve_fit(distr.pdf, x, mat[i,k,:], p0=[5,0,1000])  
                    # params, covf = curve_fit(distr.pdf, x, mat[i,k,:], p0=[50,0,50])    #init conditions: works well but not all the time
                    params, covf = curve_fit(distr.pdf, x, mat[i,k,:], p0=[20,-avg/2,std/2])  #empirical way of determining initial condition, works ok with chi2
                    
                    perr = np.sqrt(np.diag(covf))
                    if any(np.isinf(params)) or any(np.isnan(params)) or np.isinf(covf[0][0]) or any(np.isnan(perr)):
                        failed = True
                    #     raise RuntimeError("")
                    # print(f"{k},{i}: {perr}")
                    extr[i,k] = distr.ppf(extr_prob, params[0], loc=params[1], scale=params[2])
                except RuntimeError:
                    failed = True
                    
                if failed:
                    print(f"Could not determine params for a {distr_list[k]} at {k},{i}. Will just double the max load.")
                    imax = np.where(mat[i,k,:] >= thr)
                    extr[i,k] = 2.*x[imax[0][-1]] if len(imax[0])>0 else 0.0
                    params = (np.nan,0,1)

                p[i,k,0] = params[0] #can I do something better than this?
                p[i,k,1] = params[1] #can I do something better than this?
                p[i,k,2] = params[2] #can I do something better than this?

    return extr, p


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

withEXTR = True  #compute EXTREME moments 
withDEL = True  #compute DEL moments - if both are False, the lofi optimization falls back to a default WISDEM run
doLofiOptim = False  #skip lofi optimization, if you are only interested in getting the DEL and EXTR outputs (e.g. for HiFi)
nGlobalIter = 1
restartAt = 0

extremeExtrapMeth = 3
#1: just compute avg and std of the data, and rebuild a normal distribution for that
#2: try the fit function of scipy.stats to the whole data: EXPERIMENTAL, and does not seem to be using it properly
#3: curvefit the distributions to the histogramme - RECOMMENDED APPROACH

readOutputFrom = "" #results path where to get output data. I not empty, we do bypass OpenFAST execution and only postprocess files in that folder instead
readOutputFrom = mydir + os.sep + "tmp2"
#CAUTION: when specifying a readOutput, you must make sure that the modeling_option.yaml you provide actually correspond to those outputs (mostly the descrition of simulation time and IEC conditions)

fname_analysis_options_FORCED = ""

showPlots = False

# Design choice in fatigue: for how long do you size the turbine + other parameters
m_wohler = 10 #caution: also hardcoded in the definition of fatigue_channels at the top of runFAST_pywrapper 
    #TODO: could handle this better by not relying on the output of run_weis but instead rereading all the .out files with a new instance of LoadsAnalysis that uses the fatigue channels defined above
Tlife = 3600 * 24 * 365 * 20 #the design life of the turbine, in seconds (20 years)
# f_eq = 1 #rotor rotation freq is around 0.1Hz. -- THIS IS TOTALLY ARBITRARY FOR NOW
f_eq = 1/Tlife #--> RECOMMENDED SETTING
#Note on the choice of f_eq:
# - it has no influence at all on low fidelity optimization
# - for high-fidelity, it may have some. However, the formulation that uses 1/Tlife ensures that the 
#   damage constraint takes the same expression as the failure constraint. So applying the corresponding
#   DEL to the structure and checking for failure is equivalent to making sure there is no fatigue failure.
#   - Anyway, we recommend keeping f_eq<=1. We observed overstimated damage with higher values.

#==================== ======== =====================================
## Preprocessing

withDorE = withDEL or withEXTR

if not withDorE and not doLofiOptim: nGlobalIter = 0
if not withDorE or not doLofiOptim or fname_analysis_options_FORCED: nGlobalIter = 1

iec_dlc_for_del = 1.1 #hardcoded
iec_dlc_for_extr = 1.3 #hardcoded

# analysis_opt = load_yaml(fname_analysis_options)
wt_init = load_yaml(fname_wt_input)
modeling_options = load_yaml(fname_wt_input)  #initial load

#  Write the WEIS input file
analysis_options_WEIS = {}
analysis_options_WEIS["general"] = {}
analysis_options_WEIS["general"]["folder_output"] = "outputs_WEIS"
analysis_options_WEIS["general"]["fname_output"] = "DTU10MW_Madsen"

my_write_yaml(analysis_options_WEIS, fname_analysis_options_WEIS)

#  Ability to restart from a previous iteration:
restartAt = max(0,restartAt)
current_wt_input = fname_wt_input
if restartAt > 0:
    folder_wt_restart = folder_arch + os.sep + "outputs_optim" + os.sep + f"iter_{restartAt-1}"
    if not os.path.isdir(folder_wt_restart):
        raise FileNotFoundError(f"Can't restart from iter {restartAt-1} in folder {folder_wt_restart}")     
    current_wt_input = folder_wt_restart + os.sep + "blade_out.yaml"
        

#  Preparation of the channels passed to the LoadsAnalysis reader
magnitude_channels = {}
fatigue_channels = {}
fatigue_channels = {}
for i in range(1,41): #TODO: must read this number from somewhere!!
    tag = "B1N%03iMLx"%(i)
    fatigue_channels[tag] = m_wohler
    tag = "B1N%03iMLy"%(i)
    fatigue_channels[tag] = m_wohler
    tag = "B1N%03iFLz"%(i)
    fatigue_channels[tag] = m_wohler
    tag = "AB1N%03iFn"%(i)
    fatigue_channels[tag] = m_wohler
    tag = "AB1N%03iFt"%(i)
    fatigue_channels[tag] = m_wohler
    tag = "AB1N%03iFx"%(i)
    fatigue_channels[tag] = m_wohler
    tag = "AB1N%03iFy"%(i)
    fatigue_channels[tag] = m_wohler


#==================== ======== =====================================
# Unsteady loading computation from DLCs

for IGLOB in range(restartAt,nGlobalIter):
    print("\n\n\n  ============== ============== ===================\n")
    print(f"  ============== GLOBAL ITER {IGLOB} ===================\n")
    print("  ============== ============== ===================\n\n\n\n")

    # preliminary definition:
    if readOutputFrom:
        simfolder = readOutputFrom
    else:
        simfolder = mydir + os.sep + modeling_options["openfast"]["file_management"]["FAST_runDirectory"]


    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #           PHASE 1 : Compute DEL and extrapolate extreme loads
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if withDorE and not fname_analysis_options_FORCED:
        
        if not readOutputFrom:

            # Run the base simulation
            wt_opt, modeling_options, opt_options = run_weis(
                current_wt_input, fname_modeling_options, fname_analysis_options_WEIS
            )

            DELs = wt_opt['aeroelastic.DELs']

            fast_fnames = DELs.index #the names in here are the filename prefix

            fast_dlclist = wt_opt['aeroelastic.dlc_list']

            print("\n\n\n  -------------- DONE WITH WEIS ------------------\n\n\n\n")
            sys.stdout.flush()

        else:
            modeling_options = load_yaml(fname_modeling_options)
            # opt_options = load_yaml(fname_analysis_options_WEIS)

            # list all output files in the dir
            fast_fnames = []
            ls = os.listdir(readOutputFrom)
            for file in ls:
                if ".out" in file:
                    fast_fnames.append(file)

            if not fast_fnames:
                raise Warning(f"could not find any output files in the directory {readOutputFrom}")
            print(f"Will try to read the following files: {fast_fnames}")
            
            fast_fnames.sort() # sort filename list to make sure they come in the same order as they were computed

            la = LoadsAnalysis(
                outputs= fast_fnames,
                directory = readOutputFrom,
                magnitude_channels=magnitude_channels,
                fatigue_channels=fatigue_channels,
                #extreme_channels=channel_extremes,
                trim_data = (modeling_options["Level3"]["simulation"]["TStart"], modeling_options["Level3"]["simulation"]["TMax"]),
            )

            la.process_outputs(1) #TODO: MULTICORE?
            # summary_stats = la._summary_stats
            # extremes = la._extremes
            DELs = la._dels

            # Re-determine what were the DLCs run in each simulation
            fast_dlclist = []
            iec_settings = modeling_options["openfast"]["dlc_settings"]["IEC"]
            for dlc in iec_settings:
                if iec_dlc_for_del == dlc["DLC"]:
                    for i in dlc["U"]:
                        fast_dlclist.append(dlc["DLC"])
                if iec_dlc_for_extr == dlc["DLC"]:
                    for i in dlc["U"]:
                        fast_dlclist.append(dlc["DLC"])
            
        # ----------------------------------------------------------------------------------------------
        #    specific preprocessing and definitions

        # nt = len(ct[0]["B1N001FLz"])
        nx = modeling_options["WISDEM"]["RotorSE"]["n_span"]
        nx_hard = 40 #hardcoded in runFAST_pywrapper
        if nx > nx_hard: 
            raise RuntimeError("Not enough channels for DELs provisionned in runFAST_pywrapper.")

        fac = np.array([1.,1.,1.e3,1.e3,1.e3]) #multiplicator because output of AD is in N, but output of ED is in kN
        labs = ["Fn [N/m]","Ft [N/m]","MLx [kNm]","MLy [kNm]","FLz [kN]"]

        # ----------------------------------------------------------------------------------------------
        #    probability of the turbine to operate in specific conditions. 

        pp = PowerProduction(wt_init['assembly']['turbine_class'])
        iec_settings = modeling_options["openfast"]["dlc_settings"]["IEC"]
        iec_del = {}
        iec_extr = {}
        Udel_str = [] #dummy
        Uextr_str = [] #dummy
        for dlc in iec_settings:
            if iec_dlc_for_del == dlc["DLC"]:
                iec_del = dlc
                Udel_str = iec_del["U"]
            if iec_dlc_for_extr == dlc["DLC"]:
                iec_extr = dlc
                Uextr_str = iec_extr["U"]

        pj = pp.prob_WindDist([float(u) for u in Udel_str], disttype='pdf')
        pj = pj / np.sum(pj) #renormalizing so that the sum of all the velocity we simulated covers the entire life of the turbine
        #--
        # pj = np.ones(Nj) / Nj   #uniform probability instead

        pj_extr = pp.prob_WindDist([float(u) for u in Uextr_str], disttype='pdf')
        pj_extr = pj_extr / np.sum(pj_extr) #renormalizing so that the sum of all the velocity we simulated covers the entire life of the turbine

        # ----------------------------------------------------------------------------------------------
        #   reading DEL data and setting up indices
        
        # Init our lifetime DEL
        DEL_life_B1 = np.zeros([nx,5])    

        #  -- Retreive the DELstar --
        # (after removing "elapsed" from the del post_processing routine in weis)
        npDelstar = DELs.to_numpy()
        
        #duration of  time series
        Tj = modeling_options["Level3"]["simulation"]["TMax"] - modeling_options["Level3"]["simulation"]["TStart"]

        #number of time series
        Nj = len(npDelstar)


        if withDEL:
            print(f"Found {Nj} time series...")
            DELs.info()

            if len(pj)+len(pj_extr) != Nj: 
                raise Warning("Not the same number of velocities in the input yaml and in the output files.")
                #later, treat the case of DLCs other than 1.1 and 1.3

            # Indices where to find DELs for the various nodes:
            colnames = DELs.columns
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
                print("Warning: I did not find required data among time series to compute DEL! They will end up being 0.")
            else:
                print(f"Time series {jDEL} are being processed for DEL...")


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
                    jloc = j - jDEL[0]
                    DEL_life_B1[:,k] += fj[jloc] * npDelstar[jloc][ids] 
                k+=1
            DEL_life_B1 = .5 * fac * ( DEL_life_B1 / n_life_eq ) ** (1/m_wohler)

            # More processing:
            #1) switch from IEC local blade frame to "airfoil frame" with x towards TE
            tmp = DEL_life_B1[:,2].copy()
            DEL_life_B1[:,2] = -DEL_life_B1[:,3]
            DEL_life_B1[:,3] = tmp
            DEL_life_B1[:,4] = DEL_life_B1[:,4]

            print("Damage eq loads:")
            print(np.transpose(DEL_life_B1))


        # ----------------------------------------------------------------------------------------------
        # -- proceed to extreme load/gust extrapolation if requested

        if withEXTR:
            # Identify what time series correspond to extreme loads - as per IEC standard, we use ETW -- labeled DLC 1.3
            jEXTR = []
            for j in range(Nj):
                if 1.3 == fast_dlclist[j]:
                    jEXTR.append(j)  
            
            if not jEXTR:
                print("Warning: I did not find required data among time series to compute extreme loads! They will end up being 0.")
            else:
                print(f"Time series {jEXTR} are being processed for extreme loads...")

            nbins = 100
            nt = int( Tj / modeling_options["Level3"]["simulation"]["DT"] ) + 1

            # Init our extreme loads
            EXTR_distro_B1 = np.zeros([nx,5,nbins])    
            if extremeExtrapMeth ==2:
                EXTR_data_B1 = np.zeros([nx,5,nt])    
            
            #must use the same range if we want to be able to simply sum with a weight corresponding to wind probability:
            #XXX: CAUTION: this required some manual tuning, and will need retuning for another turbine...
            rng = [ (-2.e3,10.e3),
                    (-2.e3,6.e3),
                    (-8.e3,8.e3),
                    (-5.e3,2.e4),
                    (-1.e3,4.e3)] 

            for j in jEXTR:
                jloc = j - jEXTR[0]
                # Because raw data are not available as such, need to go back look into the output files:
                fname = simfolder + os.sep + fast_fnames[j]
                fulldata = myOpenFASTread(fname, addExt=modeling_options["Level3"]["simulation"]["OutFileFmt"])
        
                # i = 0
                # print(fulldata["AB1N%03iFx"%(i+1)])
                
                k = 0
                for lab in ["AB1N%03iFx","AB1N%03iFy","B1N%03iMLx","B1N%03iMLy","B1N%03iFLz"]:
                    for i in range(nx):
                        hist, bns = np.histogram(fulldata[lab%(i+1)], bins=nbins, range=rng[k])
                        EXTR_distro_B1[i,k,:] = EXTR_distro_B1[i,k,:] + hist * pj_extr[jloc]

                        if extremeExtrapMeth ==2:
                            EXTR_data_B1[i,j,:] = fulldata[lab%(i+1)]
                    k+=1

                del(fulldata)

            #normalizing the distributions
            for k in range(5):
                dx = (rng[k][1]-rng[k][0])/(nbins)
                x = np.arange(rng[k][0]+dx/2.,rng[k][1],dx)
                normFac = 1 / (nt * dx) #normalizing factor, to bring the EXTR_distro count into non-dimensional proability
                EXTR_distro_B1[:,k,:] *= normFac 

            IEC_50yr_prob = 1. - Tj / (50*3600*24*365) #=1 - 3.8e-7 for 10min sims

            #-- Assumed distr for each of the channels --
            # Note: 
            # - the longer the simulation window, the better (avoid at all cost to include the initial transient)
            # - the beam residual moments are well approximated by Gaussian
            # - the aerodynamic loads should better correspond to chi2 or weibull_min, however the fit is very sensitive to initial conditions
            # - use "normForced" as a distribution for a failsafe normal fitting (in case too many warning). This will likely overestimate the extreme loads.
            # - because of the compounded gravitational and aero loads, MLx is bimodal... not very practival for a fit! :-(
            # - use "twiceMaxForced" as a distribution for a failsafe extreme load that amounts to twice the max recorded load.
            distr = ["weibull_min","weibull_min","norm","norm","norm"]
            distr = ["chi2","chi2","chi2","chi2","chi2"]
            distr = ["chi2","chi2","twiceMaxForced","norm","norm"] #what we recommend but chi2 curve fitting may lead to oscillations in the output loading
            # distr = ["norm","norm","norm","norm","norm"] #safer from a numerical perspective
            # distr = ["normForced","normForced","normForced","normForced","normForced"]
            if extremeExtrapMeth ==1:
                #assumes only normal
                EXTR_life_B1, EXTR_distr_p = extrapolate_extremeLoads_hist(rng, EXTR_distro_B1,IEC_50yr_prob)
            elif extremeExtrapMeth ==2:
                EXTR_life_B1, EXTR_distr_p = extrapolate_extremeLoads(EXTR_data_B1, distr, IEC_50yr_prob)
            elif extremeExtrapMeth ==3:
                EXTR_life_B1, EXTR_distr_p = extrapolate_extremeLoads_curveFit(rng, EXTR_distro_B1, distr, IEC_50yr_prob)


            
            for k in range(5):
                stp = (rng[k][1]-rng[k][0])/(nbins)
                xbn = np.arange(rng[k][0]+stp/2.,rng[k][1],stp) #(bns[:-1] + bns[1:])/2.
                plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
                ss1 = plt.plot(xbn,EXTR_distro_B1[5,k,:] )
                ss2 = plt.plot(xbn,EXTR_distro_B1[15,k,:])
                ss3 = plt.plot(xbn,EXTR_distro_B1[25,k,:])
                plt.xlabel(labs[k])

                plt.plot(EXTR_life_B1[5,k] , 0, 'x' , color=ss1[0].get_color())
                plt.plot(EXTR_life_B1[15,k], 0, 'x' , color=ss2[0].get_color())
                plt.plot(EXTR_life_B1[25,k], 0, 'x' , color=ss3[0].get_color())

                dx = (rng[k][1]-rng[k][0])/(nbins)
                xx = np.arange(rng[k][0]+dx/2.,rng[k][1],dx)

                
                print(EXTR_distr_p[5,k,:])
                print(EXTR_distr_p[15,k,:])
                print(EXTR_distr_p[25,k,:])

                if "norm" in distr[k]:
                    plt.plot(xx, stats.norm.pdf(xx, loc = EXTR_distr_p[5,k,0], scale = EXTR_distr_p[5,k,1]),'--', alpha=0.6 , color=ss1[0].get_color())
                    plt.plot(xx, stats.norm.pdf(xx, loc = EXTR_distr_p[15,k,0], scale = EXTR_distr_p[15,k,1]),'--', alpha=0.6 , color=ss2[0].get_color())
                    plt.plot(xx, stats.norm.pdf(xx, loc = EXTR_distr_p[25,k,0], scale = EXTR_distr_p[25,k,1]),'--', alpha=0.6 , color=ss3[0].get_color())                        
                elif "twiceMaxForced" in distr[k]:
                    pass
                else:
                    this = getattr(stats,distr[k])
                    plt.plot(xx, this.pdf(xx, EXTR_distr_p[5,k,0], loc = EXTR_distr_p[5,k,1], scale = EXTR_distr_p[5,k,2]),'--', alpha=0.6 , color=ss1[0].get_color())
                    plt.plot(xx, this.pdf(xx, EXTR_distr_p[15,k,0], loc = EXTR_distr_p[15,k,1], scale = EXTR_distr_p[15,k,2]),'--', alpha=0.6 , color=ss2[0].get_color())
                    plt.plot(xx, this.pdf(xx, EXTR_distr_p[25,k,0], loc = EXTR_distr_p[25,k,1], scale = EXTR_distr_p[25,k,2]),'--', alpha=0.6 , color=ss3[0].get_color())
                plt.savefig(f"fit_{labs[k].split(' ')[0]}_{distr[k]}.png")
            if showPlots:
                plt.show()

            # More processing:
            #1) switch from IEC local blade frame to "airfoil frame" with x towards TE
            tmp = EXTR_life_B1[:,2].copy()
            EXTR_life_B1[:,2] = -EXTR_life_B1[:,3]
            EXTR_life_B1[:,3] = tmp
            EXTR_life_B1[:,4] = EXTR_life_B1[:,4]

            for k in range(5):
                EXTR_life_B1[:,k] *= fac[k]

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

        locs = np.linspace(0.,1.,nx)

        if withDEL:
            schema["DEL"] = {}
            schema["DEL"]["description"] = del_descr_str
            schema["DEL"]["grid_nd"] = locs.tolist() #note: the node gauges are located at np.arange(1./nx/2., 1, 1./nx) but I prefer consider that it spans then entire interval [0,1]
            schema["DEL"]["deMLx"] = DEL_life_B1[:,2].tolist()
            schema["DEL"]["deMLy"] = DEL_life_B1[:,3].tolist()
            schema["DEL"]["deFLz"] = DEL_life_B1[:,4].tolist()
        if withEXTR:
            schema["extreme"] = {}
            schema["extreme"]["description"] = extr_descr_str
            schema["extreme"]["grid_nd"] = locs.tolist()
            schema["extreme"]["deMLx"] = EXTR_life_B1[:,2].tolist()
            schema["extreme"]["deMLy"] = EXTR_life_B1[:,3].tolist()
            schema["extreme"]["deFLz"] = EXTR_life_B1[:,4].tolist()

        schema["general"]["folder_output"] = "outputs_struct_withFatigue"
        if withDEL:
            schema["constraints"]["blade"]["fatigue_spar_cap_ss"]["flag"] = True
            schema["constraints"]["blade"]["fatigue_spar_cap_ps"]["flag"] = True
            schema["constraints"]["blade"]["fatigue_spar_cap_ss"]["eq_Ncycle"] = float(n_life_eq)
            schema["constraints"]["blade"]["fatigue_spar_cap_ps"]["eq_Ncycle"] = float(n_life_eq)
            schema["constraints"]["blade"]["fatigue_spar_cap_ss"]["m_wohler"] = m_wohler
            schema["constraints"]["blade"]["fatigue_spar_cap_ps"]["m_wohler"] = m_wohler
        if withEXTR:
            schema["constraints"]["blade"]["extreme_loads_from_user_inputs"] = True

        fname_analysis_options_struct = mydir + os.sep + "analysis_options_struct_withDEL.yaml"
        my_write_yaml(schema, fname_analysis_options_struct)
        #could use write_analysis_yaml from weis instead

        schema_hifi = {}
        if withDEL:
            schema_hifi["DEL"] = {}
            schema_hifi["DEL"]["description"] = del_descr_str
            schema_hifi["DEL"]["grid_nd"] = locs.tolist()
            schema_hifi["DEL"]["Fn"] = DEL_life_B1[:,0].tolist()
            schema_hifi["DEL"]["Ft"] = DEL_life_B1[:,1].tolist()
        if withEXTR:
            schema_hifi["extreme"] = {}
            schema_hifi["extreme"]["description"] = extr_descr_str
            schema_hifi["extreme"]["grid_nd"] = locs.tolist()
            schema_hifi["extreme"]["Fn"] = EXTR_life_B1[:,0].tolist()
            schema_hifi["extreme"]["Ft"] = EXTR_life_B1[:,1].tolist()

        my_write_yaml(schema_hifi, fname_aggregatedEqLoads)


        
        for k in range(5):
            plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
            if withEXTR:
                plt.plot(locs,EXTR_life_B1[:,k], label="EXTR")
            if withDEL:
                plt.plot(locs,DEL_life_B1[:,k] , label="DEL")
            plt.ylabel(labs[k])
            plt.xlabel("r/R")
            plt.legend()
            plt.savefig(f"{labs[k].split(' ')[0]}.png")
        if showPlots:
            plt.show()
    elif fname_analysis_options_FORCED:
        #if you already postprocessed the data above, and want to do the lofi optimization
        print(f"Forced use of analysis file: {fname_analysis_options_FORCED}\nI will not check that this file contains DEL or EXTRM info. Please make sure it matches your current request.")
        fname_analysis_options_struct = fname_analysis_options_FORCED
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
    if os.path.isdir(folder_arch):
        shutil.rmtree(folder_arch,ignore_errors=True)
    os.makedirs(folder_arch)

    currFolder = f"iter_{IGLOB}"

    # shutil.copy(os.path.join(fileDirectory,file), os.path.join(workingDirectory,file))
    # shutil.copytree
    
    if os.path.isfile(fname_aggregatedEqLoads) and IGLOB==0:
        shutil.move(fname_aggregatedEqLoads,folder_arch+os.sep)
    if os.path.isfile(fname_modeling_options):
        os.system(f"cp {fname_modeling_options} {folder_arch + os.sep}")
    if os.path.isfile(fname_analysis_options_struct):
        os.system(f"cp {fname_analysis_options_struct} {folder_arch + os.sep}")
    if os.path.isdir(mydir + os.sep + "outputs_WEIS"):
        shutil.move(mydir + os.sep + "outputs_WEIS", folder_arch+ os.sep + "outputs_WEIS" + os.sep + currFolder)  
    if not readOutputFrom and os.path.isdir(simfolder): #let's not move the file if it is a path provided by the user
        shutil.move(simfolder, folder_arch + os.sep + "sim" + os.sep + currFolder)
    if os.path.isdir(mydir + os.sep + "outputs_struct_withFatigue"):
        shutil.move(mydir + os.sep + "outputs_struct_withFatigue", folder_arch + os.sep + "outputs_optim" + os.sep + currFolder)
    if os.path.isdir(mydir + os.sep + "outputs_struct"):
        shutil.move(mydir + os.sep + "outputs_struct", folder_arch + os.sep + "outputs_optim" + os.sep + currFolder)

    os.system(f"mv *.png {folder_arch + os.sep + 'figs' + os.sep + currFolder}")

    # update the path to the current optimal turbine
    current_wt_input = folder_arch + os.sep + "outputs_optim" + os.sep + currFolder + os.sep + "blade_out.yaml"

    #reset path to any precomputed data, so that if there is more iterations, we will actually do up-to-date computations
    readOutputFrom = "" 

## -- plot successive DEL --

print(f"  ============== DONE AFTER {nGlobalIter} ITER ===================\n")