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
import extrapolate_utils as exut
import matplotlib.pyplot as plt

from mpi4py import MPI

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


if __name__ == '__main__':

    #==================== DEFINITIONS  =====================================

    ## File management
    mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
    # fname_wt_input = mydir + os.sep + "IEA-10-198-RWT.yaml"
    fname_wt_input = mydir + os.sep + "Madsen2019_10_forWEIS.yaml"
    fname_wt_input = mydir + os.sep + "../09_compare_struct/Madsen2019_10_forWEIS_isotropic_IC.yaml"
    fname_modeling_options = mydir + os.sep + "modeling_options.yaml"
    fname_analysis_options = mydir + os.sep + "analysis_options_struct.yaml"
    fname_analysis_options_WEIS = mydir + os.sep + "analysis_options_WEIS.yaml"
    fname_aggregatedEqLoads = mydir + os.sep + "aggregatedEqLoads.yaml"

    folder_arch = mydir + os.sep + "results"


    withEXTR = True  #compute EXTREME moments 
    withDEL = True  #compute DEL moments - if both are False, the lofi optimization falls back to a default WISDEM run
    doLofiOptim = False  #skip lofi optimization, if you are only interested in getting the DEL and EXTR outputs (e.g. for HiFi)
    nGlobalIter = 1
    restartAt = 0

    #-Extreme load extrapolation-:
    extremeExtrapMeth = 3
    #1: statistical moment-based method: just compute avg and std of the data, and rebuild a normal distribution for that
    #2: try the fit function of scipy.stats to the whole data: EXPERIMENTAL, and does not seem to be using it properly
    #3: curvefit the distributions to the histogramme - RECOMMENDED APPROACH
    logfit = True #True: fit the log of the survival function. False: fit the pdf
    killUnder = 1E-14 #remove all values in the experimental distribution under this threshold (numerical noise)


    readOutputFrom = "" #results path where to get output data. If not empty, we do bypass OpenFAST execution and only postprocess files in that folder instead
    #CAUTION: when specifying a readOutput, you must make sure that the modeling_option.yaml you provide actually correspond to those outputs (mostly the descrition of simulation time and IEC conditions)

    fname_analysis_options_FORCED = ""

    showPlots = False
    saveExtrNpy = "extrmDistro.npz"

    # Design choice in fatigue: for how long do you size the turbine + other parameters
    m_wohler = 10 #caution: also hardcoded in the definition of fatigue_channels at the top of runFAST_pywrapper 
        #TODO: could handle this better by not relying on the output of run_weis but instead rereading all the .out files with a new instance of LoadsAnalysis that uses the fatigue channels defined above
    Textr = 3600 * 24 * 365 * 50 # return period of the extreme event, in seconds (e.g. 50yr)
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
    if MPI:
        rank = MPI.COMM_WORLD.Get_rank()
        commSize = MPI.COMM_WORLD.Get_size()
    else:
        rank = 0
        commSize = 1

    # Determine on how many threads to run the processing:
    # NUM_THREAD = int( os.environ.get('OMP_NUM_THREAD') )
    # if NUM_THREAD is None:
    #     NUM_THREAD = int( os.environ.get('SLURM_CPUS_PER_TASK') )
    # if NUM_THREAD is None:
    #     NUM_THREAD = 1
    NUM_THREAD = None


    withDorE = withDEL or withEXTR

    if not withDorE and not doLofiOptim: nGlobalIter = 0
    if not withDorE or not doLofiOptim or fname_analysis_options_FORCED: nGlobalIter = 1

    iec_dlc_for_del = 1.1 #hardcoded
    iec_dlc_for_extr = 1.3 #hardcoded

    # analysis_opt = load_yaml(fname_analysis_options)
    wt_init = load_yaml(fname_wt_input)
    modeling_options = load_yaml(fname_modeling_options)  #initial load

    #  Write the WEIS input file
    analysis_options_WEIS = {}
    analysis_options_WEIS["general"] = {}
    analysis_options_WEIS["general"]["folder_output"] = "outputs_WEIS"
    analysis_options_WEIS["general"]["fname_output"] = "DTU10MW_Madsen"

    if rank == 0:
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

    MPI.COMM_WORLD.Barrier()

    #==================== ======== =====================================
    # Initialize timers
    elapsed_sim = 0.0
    elapsed_postpro = 0.0
    elapsed_optim = 0.0

    if rank == 0:
        print(f"Walltime: {MPI.Wtime()}")

    #==================== ======== =====================================
    # Unsteady loading computation from DLCs

    for IGLOB in range(restartAt,nGlobalIter):
        if rank == 0:
            print("\n\n\n  ============== ============== ===================\n"
                + f"  ============== GLOBAL ITER {IGLOB} ===================\n"
                + "  ============== ============== ===================\n\n\n\n")
            wt = MPI.Wtime()
            wt_tot = wt
            wt_sim = wt

        # preliminary definition:
        if readOutputFrom:
            simfolder = readOutputFrom
        else:
            simfolder = mydir + os.sep + modeling_options["openfast"]["file_management"]["FAST_runDirectory"]
            if commSize >1:
                simfolder += os.sep + "rank_0"


        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #           PHASE 1 : Compute DEL and extrapolate extreme loads
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if withDorE and not fname_analysis_options_FORCED:
            
            if not readOutputFrom:

                # Run the base simulation
                wt_opt, modeling_options, opt_options = run_weis(
                    current_wt_input, fname_modeling_options, fname_analysis_options_WEIS
                )

                if rank == 0:
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

                # Sort the file list, to avoid relying on the order determined by 'ls'
                fast_fnames.sort()

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

                print(f"pCrunch: will run the analysis on {NUM_THREAD} threads.")
                la.process_outputs(cores=NUM_THREAD) 
                # summary_stats = la._summary_stats
                # extremes = la._extremes
                DELs = la._dels

                # Re-determine what were the DLCs run in each simulation
                fast_dlclist = []
                iec_settings = modeling_options["openfast"]["dlc_settings"]["IEC"]
                for dlc in iec_settings:
                    if iec_dlc_for_del == dlc["DLC"]:
                        for i in dlc["U"]:
                            for j in dlc["Seeds"]:
                                fast_dlclist.append(dlc["DLC"])
                    if iec_dlc_for_extr == dlc["DLC"]:
                        for i in dlc["U"]:
                            for j in dlc["Seeds"]:
                                fast_dlclist.append(dlc["DLC"])
                
            if rank == 0:
                wt = MPI.Wtime()
                elapsed_sim = wt - wt_sim

            # ----------------------------------------------------------------------------------------------
            # ----------------------------------------------------------------------------------------------
            #### POST-PROCESSING
            # ----------------------------------------------------------------------------------------------
            # ----------------------------------------------------------------------------------------------
            
            MPI.COMM_WORLD.Barrier()

            # only by rank 0
            if rank == 0:
                wt_postpro = wt
                print(f"{rank} done with sim")

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
                nSEEDdel  = 0 
                nSEEDextr = 0 
                for dlc in iec_settings:
                    if iec_dlc_for_del == dlc["DLC"]:
                        iec_del = dlc
                        Udel_str = iec_del["U"]
                        nSEEDdel += len(iec_del["Seeds"])
                    if iec_dlc_for_extr == dlc["DLC"]:
                        iec_extr = dlc
                        Uextr_str = iec_extr["U"]
                        nSEEDextr += len(iec_extr["Seeds"])

                pj = pp.prob_WindDist([float(u) for u in Udel_str], disttype='pdf')
                pj = pj / np.sum(pj) #renormalizing so that the sum of all the velocity we simulated covers the entire life of the turbine
                #--
                # pj = np.ones(Nj) / Nj   #uniform probability instead

                pj_extr = pp.prob_WindDist([float(u) for u in Uextr_str], disttype='pdf')
                pj_extr = pj_extr / np.sum(pj_extr) #renormalizing so that the sum of all the velocity we simulated covers the entire life of the turbine

                # ----------------------------------------------------------------------------------------------
                #   reading data and setting up indices
                
                # Init our lifetime DEL
                DEL_life_B1 = np.zeros([nx,5])    

                #  -- Retreive the DELstar --
                # (after removing "elapsed" from the del post_processing routine in weis)
                npDelstar = DELs.to_numpy()
                
                #duration of  time series
                Tj = modeling_options["Level3"]["simulation"]["TMax"] - modeling_options["Level3"]["simulation"]["TStart"]

                #number of time series
                Nj = len(npDelstar)

                # DO CHECKS
                print(f"Found {Nj} time series...")
                DELs.info()

                if len(pj)*nSEEDdel+len(pj_extr)*nSEEDextr != Nj: 
                    raise Warning("Not the same number of velocities and seeds in the input yaml and in the output files.")
                    #TODO: treat the case of DLCs other than 1.1 and 1.3, with different number of velicities and seeds


                # ----------------------------------------------------------------------------------------------
                # -- proceed to DEL aggregation if requested

                if withDEL:

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
                        #NOTE: we assume that all simulatons of a given DLC are in a row
                        #TODO: better handle this for various DLCs with different nvels and nseeds
                        for ivel in range(len(pj)): #nvels
                            for iseed in range(nSEEDdel):
                                jloc = (iseed + ivel * nSEEDdel) + jDEL[0]

                                # print(f"[{k}] v{ivel} s{iseed} loc{jloc} - {fast_fnames[jloc]} {fast_dlclist[jloc]}")

                                #average the DEL over the seeds: just as if we aggregated all the seeds
                                DEL_life_B1[:,k] += fj[ivel] * npDelstar[jloc][ids] / nSEEDdel

                        k+=1
                    DEL_life_B1 = .5 * fac * ( DEL_life_B1 / n_life_eq ) ** (1/m_wohler)

                    # summary_stats = la._summary_stats
                    # for ch in ['BldPitch1','GenSpeed',"RtAeroFxh","RotThrust","RtAeroCt","GenPwr","RotTorq","RtAeroCp"]:
                    #     print(ch)
                    #     print(summary_stats[ch])  #['mean']

                                
                    # c. More processing:
                    # -> switch from IEC local blade frame to "PRECOMP frame" with x positive towards TE
                    #    Notes: 
                    #    - ELASTODYN output is in "local coordinate system similar to the standard blade system", 
                    #       So we have x upwards and y chordwise positive towards LE. This is
                    #       confirmed by the results that give positive M in x AND y, with moment in y (flapwise) larger.
                    #       Note: this is NOT in principal elastic axes, just airfoil-aligned axes.
                    #    - WISDEM strain computation needs the moments in principal axes coordinates: 1st principal 
                    #       direction is chordwise positive towards TE, and 2 is positive upwards. This is confirmed by 
                    #       the stifness properties: EI11 is lower than EI22 (lower stifness edgewise than flapwise). 
                    #       However, the strain module processes the input by swapping x and y and by rotating to the 
                    #       principal axes. So out input should be in PRECOMP axes: x positive towards suction side, y positive towards TE.
                    #    CONCLUSION: switch from ED frame to PRECOMP frame by changing sign of My
                    # DEL_life_B1[:,2] = DEL_life_B1[:,2]
                    DEL_life_B1[:,3] = -DEL_life_B1[:,3] 
                    # DEL_life_B1[:,4] = DEL_life_B1[:,4]

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
                    dt = modeling_options["Level3"]["simulation"]["DT"]
                    nt = int( Tj / dt ) + 1

                    # Init our extreme loads
                    EXTR_distro_B1 = np.zeros([nx,5,nbins])    
                    if extremeExtrapMeth ==2:
                        EXTR_data_B1 = np.zeros([nx,5,nt])    
                    
                    #must use the same range if we want to be able to simply sum with a weight corresponding to wind probability:
                    #XXX: CAUTION: this required some manual tuning, and will need retuning for another turbine...
                    rng = [ (-2.e3,12.e3),
                            (-2.e3,6.e3),
                            (-8.e3,8.e3),
                            (-5.e3,2.e4),
                            (-1.e3,4.e3)] 


                    #TODO: better handle this for various DLCs with different nvels and nseeds
                    for ivel in range(len(pj_extr)): #nvels
                        for iseed in range(nSEEDextr): 
                            jloc = (iseed + ivel * nSEEDextr) + jEXTR[0]

                            # Because raw data are not available as such, need to go back look into the output files:
                            fname = simfolder + os.sep + fast_fnames[jloc]
                            print(fname)
                            fulldata = myOpenFASTread(fname, addExt=modeling_options["Level3"]["simulation"]["OutFileFmt"])
                        
                            k = 0
                            for lab in ["AB1N%03iFx","AB1N%03iFy","B1N%03iMLx","B1N%03iMLy","B1N%03iFLz"]:
                                # print(f"[{lab}] v{ivel} s{iseed} loc{jloc} - {fast_fnames[jloc]} {fast_dlclist[jloc]}")
                                for i in range(nx):
                                    hist, bns = np.histogram(fulldata[lab%(i+1)], bins=nbins, range=rng[k])
                                    #average the EXTRM over the seeds: just as if we aggregated all the seeds
                                    EXTR_distro_B1[i,k,:] +=  hist * pj_extr[ivel] / nSEEDextr

                                    if extremeExtrapMeth ==2:
                                        EXTR_data_B1[i,j,:] = fulldata[lab%(i+1)]
                                k+=1

                            del(fulldata)

                    #normalizing the distributions
                    for k in range(5):
                        dx = (rng[k][1]-rng[k][0])/(nbins)
                        x = np.arange(rng[k][0]+dx/2.,rng[k][1],dx)
                        # normFac = 1 / (nt * dx) #normalizing factor, to bring the EXTR_distro count into non-dimensional proability
                        # EXTR_distro_B1[:,k,:] *= normFac 
                        #--> there might be missing timesteps or anything... let's just make sure the distro sum to 1.00
                        for i in range(nx):
                            EXTR_distro_B1[i,k,:] /= np.sum(EXTR_distro_B1[i,k,:]*dx)

                    IEC_50yr_prob = 1. - dt / Textr #=return period 50yr
                    # Explanation: This is only due to how we fit the probability distro
                    #   - I normalize the histogtam by normFac, i.e. the Y axis is 1/load. The integral of the histogram gives a density of 1.0 (unitless)
                    #   - Thus, instead of computing prob with T/50yr, I do dt/50yr since I normalized already with nt.
                    #   - There is no time in this density function, except implicitely the sampling period. 
                    #   - The max load that cooresponds to a return period of 2dT has a probability of 1-dt/2dt = 0.5, that is the mean of our distribution.
                    #   - So the prob should read: 50yr corresponds to 
                    #   - What's conterintuitive to me: the smaller the dt, the larger the 50yr. load... but this might be ok: if dt increases, you have a better certainty on the distro.


                    #-- Assumed distr for each of the channels --
                    # Note: 
                    # - the longer the simulation window, the better (avoid at all cost to include the initial transient)
                    # - the beam residual moments are well approximated by Gaussian
                    # - the aerodynamic loads should better correspond to chi2 or weibull_min, however the fit is very sensitive to initial conditions
                    # - use "normForced" as a distribution for a failsafe normal fitting (in case too many warning). It reverts back to moment-based fit. This will likely overestimate the extreme loads.
                    # - because of the compounded gravitational and aero loads, MLx is bimodal... not very practival for a fit! :-(
                    # - use "twiceMaxForced" as a distribution for a failsafe extreme load that amounts to twice the max recorded load.
                    distr = ["weibull_min","weibull_min","norm","norm","norm"]
                    distr = ["chi2","chi2","chi2","chi2","chi2"]
                    distr = ["chi2","chi2","twiceMaxForced","norm","norm"] #chi2 curve fitting may lead to oscillations in the output loading
                    # distr = ["norm","norm","norm","norm","norm"] #safer from a numerical perspective
                    # distr = ["gumbel_r","gumbel_r","gumbel_r","gumbel_r","gumbel_r",]
                    # distr = ["weibull_min","weibull_min","weibull_min","weibull_min","weibull_min"]
                    # distr = ["normForced","normForced","normForced","normForced","normForced"]
                    # -- Restrict the portion of data considered for the fit (keep the tail only) ---------
                    truncThr = None #no restriction

                    # new recommended setup:
                    distr = ["norm","norm","twiceMaxForced","norm","norm"] 
                    # truncThr = [0.5,1.0,None,0.5,0.5] #recommend using None if logfit=true
                    # ------------
                    if extremeExtrapMeth ==1:
                        #assumes only normal
                        EXTR_life_B1, EXTR_distr_p = exut.extrapolate_extremeLoads_hist(rng, EXTR_distro_B1,IEC_50yr_prob)
                    elif extremeExtrapMeth ==2:
                        EXTR_life_B1, EXTR_distr_p = exut.extrapolate_extremeLoads(EXTR_data_B1, distr, IEC_50yr_prob)
                    elif extremeExtrapMeth ==3:
                        EXTR_life_B1, EXTR_distr_p = exut.extrapolate_extremeLoads_curveFit(rng, EXTR_distro_B1, distr, IEC_50yr_prob, truncThr=truncThr, logfit=logfit, killUnder=killUnder)

                    # ------------ TIMIMGS ------------    
                    wt = MPI.Wtime()
                    elapsed_postpro = wt - wt_postpro

                    # ------------ DUMPING RESULTS ------------    
                    if saveExtrNpy:
                        np.savez(saveExtrNpy, rng=rng, nbins=nbins, EXTR_life_B1=EXTR_life_B1, EXTR_distr_p=EXTR_distr_p, EXTR_distro_B1=EXTR_distro_B1, distr=distr, dt=dt)
                
                    # ------------ PLOTTING ------------    
                    for k in range(5):
                        stp = (rng[k][1]-rng[k][0])/(nbins)
                        xbn = np.arange(rng[k][0]+stp/2.,rng[k][1],stp) #(bns[:-1] + bns[1:])/2.
                        dx = (rng[k][1]-rng[k][0])/(nbins)
                        
                        f1,ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
                        ax1.set_xlabel(labs[k])
                        
                        ss1 = ax1.plot(xbn,EXTR_distro_B1[5,k,:] )
                        ss2 = ax1.plot(xbn,EXTR_distro_B1[15,k,:])
                        ss3 = ax1.plot(xbn,EXTR_distro_B1[25,k,:])
                        c1 = ss1[0].get_color()
                        c2 = ss2[0].get_color()
                        c3 = ss3[0].get_color()
                        ax1.plot(EXTR_life_B1[5,k] , 0, 'x' , color=c1)
                        ax1.plot(EXTR_life_B1[15,k], 0, 'x' , color=c2)
                        ax1.plot(EXTR_life_B1[25,k], 0, 'x' , color=c3)

                        f2,ax2 = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
                        ax2.set_yscale('log')
                        ax2.set_xlabel(labs[k])
                        ax2.set_ylim([ (1.-IEC_50yr_prob)/2. , 2.])                

                        dsf1= 1.-np.cumsum(EXTR_distro_B1[5,k,:] )*dx 
                        dsf2= 1.-np.cumsum(EXTR_distro_B1[15,k,:])*dx 
                        dsf3= 1.-np.cumsum(EXTR_distro_B1[25,k,:])*dx 
                        dsf1[(dsf1>=1.-1E-16) | (dsf1<=1E-16)] = np.nan
                        dsf2[(dsf2>=1.-1E-16) | (dsf2<=1E-16)] = np.nan
                        dsf3[(dsf3>=1.-1E-16) | (dsf3<=1E-16)] = np.nan
                        ax2.plot(xbn,dsf1)
                        ax2.plot(xbn,dsf2)
                        ax2.plot(xbn,dsf3)
                        ax2.plot(EXTR_life_B1[5,k] , 1.-IEC_50yr_prob, 'x' , color=c1)
                        ax2.plot(EXTR_life_B1[15,k], 1.-IEC_50yr_prob, 'x' , color=c2)
                        ax2.plot(EXTR_life_B1[25,k], 1.-IEC_50yr_prob, 'x' , color=c3)
                        
                        

                        xx = np.arange(rng[k][0]+dx/2.,rng[k][1],dx)

                        
                        print(EXTR_distr_p[5,k,:])
                        print(EXTR_distr_p[15,k,:])
                        print(EXTR_distr_p[25,k,:])

                        if "twiceMaxForced" in distr[k]:
                            pass
                        elif "normForced" in distr[k]:
                            ax1.plot(xx, stats.norm.pdf(xx, loc = EXTR_distr_p[5,k,0], scale = EXTR_distr_p[5,k,1]),'--', alpha=0.6 , color=c1)
                            ax1.plot(xx, stats.norm.pdf(xx, loc = EXTR_distr_p[15,k,0], scale = EXTR_distr_p[15,k,1]),'--', alpha=0.6 , color=c2)
                            ax1.plot(xx, stats.norm.pdf(xx, loc = EXTR_distr_p[25,k,0], scale = EXTR_distr_p[25,k,1]),'--', alpha=0.6 , color=c3)                        

                            ax2.plot(xx, stats.norm.sf(xx, loc = EXTR_distr_p[5,k,0], scale = EXTR_distr_p[5,k,1]),'--', alpha=0.6 , color=c1)
                            ax2.plot(xx, stats.norm.sf(xx, loc = EXTR_distr_p[15,k,0], scale = EXTR_distr_p[15,k,1]),'--', alpha=0.6 , color=c2)
                            ax2.plot(xx, stats.norm.sf(xx, loc = EXTR_distr_p[25,k,0], scale = EXTR_distr_p[25,k,1]),'--', alpha=0.6 , color=c3) 
                        elif "norm" in distr[k] or "gumbel" in distr[k]: #2params models
                            this = getattr(stats,distr[k])
                            ax1.plot(xx, this.pdf(xx, loc = EXTR_distr_p[5,k,0], scale = EXTR_distr_p[5,k,1]),'--', alpha=0.6 , color=c1)
                            ax1.plot(xx, this.pdf(xx, loc = EXTR_distr_p[15,k,0], scale = EXTR_distr_p[15,k,1]),'--', alpha=0.6 , color=c2)
                            ax1.plot(xx, this.pdf(xx, loc = EXTR_distr_p[25,k,0], scale = EXTR_distr_p[25,k,1]),'--', alpha=0.6 , color=c3)                        

                            ax2.plot(xx, this.sf(xx, loc = EXTR_distr_p[5,k,0], scale = EXTR_distr_p[5,k,1]),'--', alpha=0.6 , color=c1)
                            ax2.plot(xx, this.sf(xx, loc = EXTR_distr_p[15,k,0], scale = EXTR_distr_p[15,k,1]),'--', alpha=0.6 , color=c2)
                            ax2.plot(xx, this.sf(xx, loc = EXTR_distr_p[25,k,0], scale = EXTR_distr_p[25,k,1]),'--', alpha=0.6 , color=c3) 
                        else: #3params models
                            this = getattr(stats,distr[k])
                            ax1.plot(xx, this.pdf(xx, EXTR_distr_p[5,k,0], loc = EXTR_distr_p[5,k,1], scale = EXTR_distr_p[5,k,2]),'--', alpha=0.6 , color=c1)
                            ax1.plot(xx, this.pdf(xx, EXTR_distr_p[15,k,0], loc = EXTR_distr_p[15,k,1], scale = EXTR_distr_p[15,k,2]),'--', alpha=0.6 , color=c2)
                            ax1.plot(xx, this.pdf(xx, EXTR_distr_p[25,k,0], loc = EXTR_distr_p[25,k,1], scale = EXTR_distr_p[25,k,2]),'--', alpha=0.6 , color=c3)

                            ax2.plot(xx, this.sf(xx, EXTR_distr_p[5,k,0], loc = EXTR_distr_p[5,k,1], scale = EXTR_distr_p[5,k,2]),'--', alpha=0.6 , color=c1)
                            ax2.plot(xx, this.sf(xx, EXTR_distr_p[15,k,0], loc = EXTR_distr_p[15,k,1], scale = EXTR_distr_p[15,k,2]),'--', alpha=0.6 , color=c2)
                            ax2.plot(xx, this.sf(xx, EXTR_distr_p[25,k,0], loc = EXTR_distr_p[25,k,1], scale = EXTR_distr_p[25,k,2]),'--', alpha=0.6 , color=c3)
                        f1.savefig(f"fit_{labs[k].split(' ')[0]}_{distr[k]}.png")
                        f2.savefig(f"fit_sf_{labs[k].split(' ')[0]}_{distr[k]}.png")
                    if showPlots:
                        plt.show()

                    # More processing:
                    #1) switch from IEC local blade frame to "PRECOMP frame" with y positive towards TE
                    # EXTR_life_B1[:,2] = EXTR_life_B1[:,2]
                    EXTR_life_B1[:,3] = -EXTR_life_B1[:,3]
                    # EXTR_life_B1[:,4] = EXTR_life_B1[:,4]


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

                locs = np.linspace(0.,1.,nx) #XXX Is this right? Shoule we obtain it from rotorse instead?

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


        MPI.COMM_WORLD.Barrier()

        if rank == 0:
            wt = MPI.Wtime()
            wt_optim = wt

        # +++++++++++++++++++++++++++++++++++++++
        #           PHASE 2 : Optimize
        # +++++++++++++++++++++++++++++++++++++++
        if doLofiOptim:
            # Let's use the most up-to-date turbine as a starting point:
            wt_opt, analysis_options, opt_options = run_wisdem(current_wt_input, fname_modeling_options, fname_analysis_options_struct)

            print("\n\n\n  -------------- DONE WITH WISDEM ------------------\n\n\n\n")    

        if rank == 0:
            wt = MPI.Wtime()
            elapsed_optim = wt - wt_optim
            
        # +++++++++++++++++++++++++++++++++++++++
        #           PHASE 3 : book keeping
        # +++++++++++++++++++++++++++++++++++++++
        currFolder = f"iter_{IGLOB}"

        if rank == 0:
            if IGLOB==0:
                if os.path.isdir(folder_arch):
                    shutil.rmtree(folder_arch,ignore_errors=True)
                os.makedirs(folder_arch)

            # shutil.copy(os.path.join(fileDirectory,file), os.path.join(workingDirectory,file))
            # shutil.copytree
            
            if os.path.isfile(fname_aggregatedEqLoads) and IGLOB==0:
                # shutil.move(fname_aggregatedEqLoads,folder_arch+os.sep)
                os.system(f"mv {fname_aggregatedEqLoads} {folder_arch+os.sep}")
            if os.path.isfile(fname_modeling_options):
                os.system(f"cp {fname_modeling_options} {folder_arch + os.sep}")
            if os.path.isfile(fname_analysis_options_struct):
                os.system(f"cp {fname_analysis_options_struct} {folder_arch + os.sep}")
            if os.path.isdir(mydir + os.sep + "outputs_WEIS"):
                os.system(f"mkdir {folder_arch + os.sep + 'outputs_WEIS'}")
                # shutil.move(mydir + os.sep + "outputs_WEIS", folder_arch+ os.sep + "outputs_WEIS" + os.sep + currFolder + os.sep)  
                os.system(f"mv {mydir + os.sep + 'outputs_WEIS'} {folder_arch+ os.sep + 'outputs_WEIS' + os.sep + currFolder}")  
            if not readOutputFrom and os.path.isdir(simfolder): #let's not move the file if it is a path provided by the user
                os.system(f"mkdir {folder_arch + os.sep + 'sim'}")
                # shutil.move(simfolder, folder_arch + os.sep + "sim" + os.sep + currFolder + os.sep, copy_function = shutil.copytree)
                os.system(f"mv {simfolder} {folder_arch + os.sep + 'sim' + os.sep + currFolder}")
            if os.path.isdir(mydir + os.sep + "outputs_struct_withFatigue"):
                os.system(f"mkdir {folder_arch + os.sep + 'outputs_optim'}")
                # shutil.move(mydir + os.sep + "outputs_struct_withFatigue", folder_arch + os.sep + "outputs_optim" + os.sep + currFolder)
                os.system(f"mv {mydir + os.sep + 'outputs_struct_withFatigue'}  {folder_arch + os.sep + 'outputs_optim' + os.sep + currFolder}")
            if os.path.isdir(mydir + os.sep + "outputs_struct"):
                os.system(f"mkdir {folder_arch + os.sep + 'outputs_optim'}")
                # shutil.move(mydir + os.sep + "outputs_struct", folder_arch + os.sep + "outputs_optim" + os.sep + currFolder)
                os.system(f"mv {mydir + os.sep + 'outputs_struct'} {folder_arch + os.sep + 'outputs_optim' + os.sep + currFolder}")
            if saveExtrNpy and os.path.isfile(saveExtrNpy):
                os.system(f"cp {saveExtrNpy} {folder_arch + os.sep}")        

            figdir = folder_arch + os.sep + 'figs' 
            if not os.path.isdir(figdir):
                os.makedirs(figdir)
            thisfdir = figdir + os.sep + currFolder
            if not os.path.isdir(thisfdir):
                os.makedirs(thisfdir)
            os.system(f"mv *.png {thisfdir}")

            # --- TIMINGS --
            wt = MPI.Wtime()
            elapsed_tot = wt - wt_tot

            with open(folder_arch + os.sep + "timings.txt", "a") as file:
                file.write('%8.16E, %8.16E, %8.16E, %8.16E, %8.16E\n' % (wt, elapsed_tot, elapsed_sim, elapsed_postpro, elapsed_optim) )

        # update the path to the current optimal turbine
        current_wt_input = folder_arch + os.sep + "outputs_optim" + os.sep + currFolder + os.sep + "blade_out.yaml"

        #reset path to any precomputed data, so that if there is more iterations, we will actually do up-to-date computations
        readOutputFrom = "" 

    ## -- plot successive DEL --

    print(f"  ============== DONE AFTER {nGlobalIter} ITER ===================\n")