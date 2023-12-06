import os
import yaml, copy

from wisdem import run_wisdem
from weis.glue_code.runWEIS import run_weis
from wisdem.inputs import load_yaml, write_yaml #, validate_without_defaults, validate_with_defaults, simple_types
from pCrunch import PowerProduction, LoadsAnalysis, FatigueParams
from pCrunch.io import OpenFASTAscii, OpenFASTBinary#, OpenFASTOutput
from weis.dlc_driver.dlc_generator    import DLCGenerator
from weis.inputs import load_modeling_yaml
import XtrFat.processingTools.extrapolate_utils as exut

import sys, shutil
import numpy as np
from scipy import stats, interpolate as scItrp
from time import time
import matplotlib.pyplot as plt

from wisdem.commonse.mpi_tools import MPI

# ---------------------
#TODO: doc
def my_write_yaml(instance, foutput):
    if os.path.isfile(foutput):
        print(f"File {foutput} already exists... replacing it.")
        os.remove(foutput)
    # Write yaml with updated values
    with open(foutput, "w", encoding="utf-8") as f:
        yaml.dump(instance, f)

#TODO: doc
def myOpenFASTread(fname,addExt=0, **kwargs):
    if addExt>0 and not ".out" in fname:
        ext = ".outb"
        if  addExt == 1: ext = ".out"
        fname += ext #appending the right extension
    try:
        output = OpenFASTAscii(fname, **kwargs) #magnitude_channels=self._mc
        output.read()

    except UnicodeDecodeError:
        output = OpenFASTBinary(fname, **kwargs) #magnitude_channels=self._mc
        output.read()

    return output

#TODO: doc
# This is mostly a copy-paste from aeroelasticse/openmdao_openfast>run_FAST
def redo_dlc_generator(modopt,wt):

    DLCs = modopt['DLC_driver']['DLCs']
    # Initialize the DLC generator
    cut_in  = float(wt['control']['supervisory']['Vin'])
    cut_out = float(wt['control']['supervisory']['Vout'])
    rated = 10 #DUMMY: does not matter because not used in DLCGenerator
    ws_class = wt['assembly']['turbine_class']
    wt_class = wt['assembly']['turbulence_class']
    hub_height = float(wt['assembly']['hub_height'])
    # rotorD = float(wt['assembly']['rotor_diameter'])
    PLExp = float(wt['environment']['shear_exp'])

    fix_wind_seeds = modopt['DLC_driver']['fix_wind_seeds']
    fix_wave_seeds = modopt['DLC_driver']['fix_wave_seeds']
    metocean = modopt['DLC_driver']['metocean_conditions']
    dlc_generator = DLCGenerator(cut_in, cut_out, rated, ws_class, wt_class, fix_wind_seeds, fix_wave_seeds, metocean)
    # Generate cases from user inputs
    for i_DLC in range(len(DLCs)):
        DLCopt = DLCs[i_DLC]
        dlc_generator.generate(DLCopt['DLC'], DLCopt)

    for i_case in range(dlc_generator.n_cases):
        if dlc_generator.cases[i_case].turbulent_wind:
            # Assign values common to all DLCs
            # Wind turbulence class
            dlc_generator.cases[i_case].IECturbc = wt_class
            # Reference height for wind speed
            dlc_generator.cases[i_case].RefHt = hub_height
            # Center of wind grid (TurbSim confusingly calls it HubHt)
            dlc_generator.cases[i_case].HubHt = hub_height
            # Height of wind grid, it stops 1 mm above the ground
            dlc_generator.cases[i_case].GridHeight = 2. * hub_height - 1.e-3
            # If OLAF is called, make wind grid high and big
            if modopt['Level3']['AeroDyn']['WakeMod'] == 3:
                dlc_generator.cases[i_case].HubHt *= 3.
                dlc_generator.cases[i_case].GridHeight *= 3.
            # Width of wind grid, same of height
            dlc_generator.cases[i_case].GridWidth = dlc_generator.cases[i_case].GridHeight
            # Power law exponent of wind shear
            dlc_generator.cases[i_case].PLExp = PLExp
            # Length of wind grids
            dlc_generator.cases[i_case].AnalysisTime = dlc_generator.cases[i_case].analysis_time + dlc_generator.cases[i_case].transient_time

    return dlc_generator

def mytime():
    if MPI:
        return MPI.Wtime()
    else:
        return time() 
# ---------------------


TWENTY_YEARS = 3600 * 24 * 365 * 20

FIFTY_YEARS = 3600 * 24 * 365 * 50

# total range over which we bin, for each quantity monitored.
#   The range should span the entire range of values we can get from the simultaion (from min to max)
RNG_DEFAULT = [ (-2.e4,2.e4), #Fx [N/m]
        (-2.e4,2.e4),  #Fy [N/m]
        (-5.e4,5.e4),  #MLx [kNm]
        (-5.e4,5.e4),  #MLy [kNm]
        (-5.e3,5.e3),  #FLz [kN]
        (-6.e-3,6.e-3),  #StrainU [-]
        (-6.e-3,6.e-3),  #StrainL [-]
        (-6.e-3,6.e-3),  #StrainTE [-]
        ]


"""
XtrFat uses WEIS to perform a series of OpenFast simulations according to a modeling and analysis file,
then processes the output to extract the extrapolated extreme loads and the aggregated life fatigue loads.
These loads are dumped under various formats for later processing and reuse.
Optionally, an optimization can be carried out at the end which can include the fatigue/extreme loads,
assuming they won't change much during the optimization. The whole sequence can be iterated over, in what
is here referred to as the outer(most) loop.

#TODO: ARGUMENTS!!!
See also the driver under examples_BYU/07_test_iterateDEL

# rng_modulation_x  
# rng_modulation_val
# Since this range may be too large in some regions of the blades (e.g. at the tip, where the loads are smaller)
#  we allow to modulate the range. rng will be scaled by a factor interpolated from rng_modulation based on
#  the spanwise location.


#-Extreme load extrapolation-:
#0: just take the max of the observed loads during the timeseries
#1: statistical moment-based method: just compute avg and std of the data, and rebuild a normal distribution for that
#2: try the fit function of scipy.stats to the whole data: EXPERIMENTAL, and does not seem to be using it properly
#3: curvefit the distributions to the histogramme - RECOMMENDED APPROACH


#-- Assumed distr for each of the channels --
# Note: 
# - the longer the simulation window, the better (avoid at all cost to include the initial transient)
# - the beam residual moments are well approximated by Gaussian
# - the aerodynamic loads should better correspond to chi2 or weibull_min, however the fit is very sensitive to initial conditions
# - use "normForced" as a distribution for a failsafe normal fitting (in case too many warning). It reverts back to moment-based fit. This will likely overestimate the extreme loads.
# - because of the compounded gravitational and aero loads, MLx is bimodal... not very practival for a fit! :-(
# - use "twiceMaxForced" as a distribution for a failsafe extreme load that amounts to twice the max recorded load.

#NOTE:
# [norm and 1.0] seems to be working well for bimodal distributions. That's the case for MLx,FLz,StrainTE.
# Fn and Ft are skewed distributions, but their tail is actually well fitted by a normal. Ft could work with a gumbel_r
# FLz has a super weird tri-modal shape
# MLx is super symmetric wrt 0
# MLy is super weird: the distribution is towards >0 but the tail is actually long towards the negative numbers, leading to an overall <0 extreme load...!
#NOTE general:
# weibull is a good distribution but it's not easy to use it for both left and right tails since it's skewed...
# norm is definitely the easiest to use and gives the best fits anyway
                        

                        

#Note on the choice of f_eq:
# - it has no influence at all on low fidelity optimization
# - for high-fidelity, it may have some. However, the formulation that uses 1/Tlife ensures that the 
#   damage constraint takes the same expression as the failure constraint. So applying the corresponding
#   DEL to the structure and checking for failure is equivalent to making sure there is no fatigue failure.
#   - Anyway, we recommend keeping f_eq<=1. We observed overstimated damage with higher values.

"""
def XtrFat(
    #folder and files
    mydir, #folder where to find all the input files ()
    fname_wt_input, #input file
    fname_modeling_options, #input file
    fname_analysis_options, #input file
    fname_analysis_options_WEIS = "analysis_options_WEIS.yaml", #output file
    readOutputFrom = "",
    folder_arch='./results', #folder where to archive all the results
    fname_aggregatedEqLoads='aggregatedAeroLoads.yaml', #output file
    #extreme:
    withEXTR = True,  #compute EXTREME moments 
    nbins = 1000, # total number of bins
    rng = RNG_DEFAULT,
    Textr = FIFTY_YEARS,
    extremeExtrapMeth = 3,
    distr = ["norm",] *8,
    truncThr = None, #no restriction
    logfit = True,  #True: fit the log of the survival function. False: fit the pdf
    killUnder = 1E-14, #remove all values in the experimental distribution under this threshold (numerical noise)
    rng_modulation_x = [0,1], 
    rng_modulation_val = [[[1,]*2,]*8],
    saveExtrNpy = "extrmDistro.npz",
    dontAggregateExtreme = False,
    #fatigue:
    withDEL = True,  #compute DEL moments - if both are False, the lofi optimization falls back to a default WISDEM run
    m_wohler = 10,
    Tlife = TWENTY_YEARS,
    f_eq = 1/TWENTY_YEARS,
    #driving:
    doLofiOptim = False, #skip lofi optimization, if you are only interested in getting the DEL and EXTR outputs (e.g. for HiFi)
    fname_analysis_options_FORCED="", #input file (to jump to lofi optim directly)
    nGlobalIter = 1,
    restartAt = 0,
    showPlots = False,
):
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
    NUM_THREAD = 1


    withDorE = withDEL or withEXTR

    if not withDorE and not doLofiOptim: nGlobalIter = 0
    if not withDorE or not doLofiOptim or fname_analysis_options_FORCED: nGlobalIter = 1

    iec_dlc_for_fat = [1.2,] #Note that it is actually DLC1.2, but it uses the same wind profile as 1.1
    if dontAggregateExtreme:
        iec_dlc_for_extr = np.array([1.1,1.3,1.4,1.5,6.1,6.3]) #List of all DLCs to be considered for extreme loads
    else:
        iec_dlc_for_extr = np.array([1.3,]) #List of all DLCs to be considered for extreme loads
        #TODO: handle the aggregation of different DLCs for extreme loads (or just consider them as separate cases and have 1extreme loading for 1DLC)

    # Associate a specific extrapolation method per DLC
    iec_dlc_meth = {}
    iec_dlc_meth["1.1"] = extremeExtrapMeth
    iec_dlc_meth["1.2"] = 0 
    iec_dlc_meth["1.3"] = extremeExtrapMeth
    iec_dlc_meth["1.4"] = 0
    iec_dlc_meth["1.5"] = 0
    iec_dlc_meth["6.1"] = 0 #extremeExtrapMeth
    iec_dlc_meth["6.3"] = 0 #extremeExtrapMeth



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
            
    if MPI:
        MPI.COMM_WORLD.Barrier()

    fname_fatMach = 'dummy'
    fname_extrMach = 'dummy'

    #==================== ======== =====================================
    # Initialize timers
    elapsed_sim = 0.0
    elapsed_postpro = 0.0
    elapsed_optim = 0.0

    if rank == 0:
        print(f"Walltime: {mytime()}")

    #==================== ======== =====================================
    # Unsteady loading computation from DLCs

    for IGLOB in range(restartAt,nGlobalIter):
        if rank == 0:
            print("\n\n\n  ============== ============== ===================\n"
                + f"  ============== GLOBAL ITER {IGLOB} ===================\n"
                + "  ============== ============== ===================\n\n\n\n")
            wt = mytime()
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

                    # fast_dlclist = wt_opt['aeroelastic.dlc_list'] # details simulation parameters for all cases run. -> no need for it if we have dlc_generator

                    DLCs = wt_opt['aeroelastic.dlc_generator'].to_dict() # details dlc parameters for all cases run. We need this to link back cases and DLCs.

                # modeling_options['DLC_driver']['n_cases']

                # Retreiving some channel information
                combili_channels = load_yaml( simfolder + os.sep + 'extra' + os.sep + 'combili_channels.yaml')
                n_span = combili_channels["n_span"]
                combili_channels.pop("n_span")

                print("\n\n\n  -------------- DONE WITH WEIS ------------------\n\n\n\n")
                sys.stdout.flush()

            else:
                modeling_options = load_modeling_yaml(fname_modeling_options) #re-read the modopts 
                # opt_options = load_yaml(fname_analysis_options_WEIS)

                # list all output files in the dir
                fast_fnames = []
                ls = os.listdir(readOutputFrom)
                for file in ls:
                    if ".out" in file:
                        fast_fnames.append(file)

                # Sort the file list, to avoid relying on the order determined by 'ls' and make sure they come in the same order as they were computed
                fast_fnames.sort()

                if not fast_fnames:
                    raise Warning(f"could not find any output files in the directory {readOutputFrom}")
                print(f"Will try to read the following files: {fast_fnames}")
                
                #  Preparation of the channels passed to the LoadsAnalysis reader
                magnitude_channels = {}
                fatigue_channels = {}
                                
                combili_channels = load_yaml( simfolder + os.sep + 'extra' + os.sep + 'combili_channels.yaml')
                n_span = combili_channels["n_span"]
                combili_channels.pop("n_span")
                Sult = 3500.e-6 #HARDCODED. Should come from something like analysis_opts["constraints"]["blade"]["strains_spar_cap_ss"]["max"]

                for k in combili_channels:
                    blade_spar_strain = FatigueParams(slope=m_wohler, DELstar=True, load2stress=1.0, ult_stress=Sult) 
                    # blade_spar_strain.load2stress = inputs[...] #can do that if needed, or from modopts?
                    fatigue_channels[k] = blade_spar_strain

                for i in range(1,n_span+1):
                    # for u in ['U','L']:
                    #     fatigue_channels[f'BladeSpar{u}_Strain_Stn{i+1}'] = blade_spar_strain
                    
                    tag = "B1N%03iMLx"%(i)
                    fatigue_channels[tag] = FatigueParams(slope=m_wohler,DELstar=True, load2stress=0.0)
                    tag = "B1N%03iMLy"%(i)
                    fatigue_channels[tag] = FatigueParams(slope=m_wohler,DELstar=True, load2stress=0.0)
                    tag = "B1N%03iFLz"%(i)
                    fatigue_channels[tag] = FatigueParams(slope=m_wohler,DELstar=True, load2stress=0.0)
                    tag = "AB1N%03iFn"%(i)
                    fatigue_channels[tag] = FatigueParams(slope=m_wohler,DELstar=True, load2stress=0.0)
                    tag = "AB1N%03iFt"%(i)
                    fatigue_channels[tag] = FatigueParams(slope=m_wohler,DELstar=True, load2stress=0.0)
                    tag = "AB1N%03iFx"%(i)
                    fatigue_channels[tag] = FatigueParams(slope=m_wohler,DELstar=True, load2stress=0.0)
                    tag = "AB1N%03iFy"%(i)
                    fatigue_channels[tag] = FatigueParams(slope=m_wohler,DELstar=True, load2stress=0.0)

                la = LoadsAnalysis(
                    outputs= fast_fnames,
                    directory = readOutputFrom,
                    magnitude_channels=magnitude_channels,
                    combili_channels=combili_channels, #they have the right units to work with kN and kNm
                    fatigue_channels=fatigue_channels,
                    #extreme_channels=channel_extremes,
                    # trim_data = (modeling_options["Level3"]["simulation"]["TStart"], modeling_options["Level3"]["simulation"]["TMax"]), #trim of data unnecessary since we only saved meaningful portion
                )

                print(f"pCrunch: will run the analysis on {NUM_THREAD} threads.")
                la.process_outputs(cores=NUM_THREAD, goodman_correction=modeling_options['General']['goodman_correction']) 
                # summary_stats = la._summary_stats
                # extremes = la._extremes
                DELs = la._dels

                DLCs = redo_dlc_generator(modeling_options, wt_init).to_dict()
                #Note: the turbine could have changed across global iterations but not the properties that we need in that function

                # # ----------------------------------------------------------------------------------------------
                # #  ------ Run a simple WISDEM evaluation to fill the wt_opt structure and have access to openMDAO structures

                # #  Modify the analysis option to do a dry run, just so that we create the openMDAO structures.
                # fname_analysis_options_dry = readOutputFrom + os.sep + "analysis_options_dry.yaml"
                # analysis_opt = load_yaml(fname_analysis_options)
                # amalysis_options_dry = copy.deepcopy(analysis_opt)
                # amalysis_options_dry["driver"]["optimization"]["flag"] = False #turn off optimization, so that we only do a simple evaluation
                # if rank == 0:
                #     my_write_yaml(amalysis_options_dry, fname_analysis_options_dry)

                # # Dry-run wisdem
                # wt_opt, _, _ = run_wisdem(current_wt_input, fname_modeling_options, fname_analysis_options_dry)

                # print("\n\n\n  -------------- DONE RELOADING ------------------\n\n\n\n")

            if rank == 0:
                wt = mytime()
                elapsed_sim = wt - wt_sim

            # ----------------------------------------------------------------------------------------------
            # ----------------------------------------------------------------------------------------------
            #### POST-PROCESSING
            # ----------------------------------------------------------------------------------------------
            # ----------------------------------------------------------------------------------------------
            
            if MPI:
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

                fac = np.array([1.,1.,1.e3,1.e3,1.e3, 1.,1.,1.]) #multiplicator because output of AD is in N, but output of ED is in kN
                # for the strain, Mx,My,Fz are in kN(m) but the combili factors include a 10^3, so that the output of the channel is indeed a strain
                labs = ["Fn [N/m]","Ft [N/m]","MLx (lead lag) [Nm]","MLy (flap, +TE) [Nm]","FLz [N]","SparStrainU [-]","SparStrainL [-]","SparStrainTE [-]"]
                labs_Lt = ["MxtU [Nm]","MytU [Nm]","FztU [N]", "MxtL [Nm]","MytL [Nm]","FztL [N]"]
                
                n_processed = len(labs) #number of quantities processed

                # ----------------------------------------------------------------------------------------------
                #   reading data and setting up indices
                
                # Init our lifetime DEL
                DEL_life_B1 = np.zeros([nx,n_processed])    
                Ltilde_life_B1 = np.zeros([nx,6])    

                #  -- Retreive the DELstar --
                # (after removing "elapsed" from the del post_processing routine in weis)
                npDelstar = DELs.to_numpy()
                
                #total number of time series
                Nj = len(npDelstar) #= len(DLCs)
                
                #number of time series in common (fatigue and extreme)
                Ncommon = 0

                print(f"Found {Nj} time series...")
                DELs.info()

                # ----------------------------------------------------------------------------------------------
                #    Processing all the cases that we get, and determine which are which in terms of DLCs

                pp = PowerProduction(wt_init['assembly']['turbine_class'])

                DLCs_fat = {}
                DLCs_extr = {}

                nTotFatSim = 0
                nTotExtrSim = 0

                il = -1
                for dlc in DLCs:
                    il += 1
                    label = dlc["label"]

                    # Match all DLCs that we wish to consider for fatigue, and assign a processing method
                    if  float( label ) in iec_dlc_for_fat:

                        if not label in DLCs_fat:
                            DLCs_fat[label] = {}
                            DLCs_fat[label]['U'] = []
                            DLCs_fat[label]['idx'] = []
                            DLCs_fat[label]['nsims'] = 0
                            DLCs_fat[label]['Tsim'] = dlc["analysis_time"] 

                        if not float( dlc["URef"] ) in DLCs_fat[label]['U']:
                            DLCs_fat[label]['U'].append(float( dlc["URef"] ) )

                        DLCs_fat[label]['idx'].append(il)
                        DLCs_fat[label]['nsims'] += 1
                        nTotFatSim += 1
                        
                    # Match all DLCs that we wish to consider for extreme, and assign a processing method
                    if  float( label ) in iec_dlc_for_extr:

                        if not label in DLCs_extr:
                            DLCs_extr[label] = {}
                            DLCs_extr[label]['U'] = []
                            DLCs_extr[label]['idx'] = []
                            DLCs_extr[label]['nsims'] = 0
                            DLCs_extr[label]['Tsim'] = dlc["analysis_time"]                             

                        if not float( dlc["URef"] ) in DLCs_extr[label]['U']:
                            DLCs_extr[label]['U'].append(float( dlc["URef"] ) )

                        DLCs_extr[label]['idx'].append(il)
                        DLCs_extr[label]['nsims'] += 1
                        nTotExtrSim += 1

                    # If one simulation is used both for fatigue and extreme loads, we just make the cound right.
                    if  float( label ) in iec_dlc_for_fat and float( label ) in iec_dlc_for_extr:
                        Ncommon += 1


                if nTotFatSim+nTotExtrSim != Nj+Ncommon: 
                    raise Warning("Not the same number of velocities and seeds in the input yaml and in the output files: %i vs %i." % (nTotFatSim+nTotExtrSim,Nj))

                # ----------------------------------------------------------------------------------------------
                # -- proceed to DEL aggregation if requested

                if withDEL and not DLCs_fat:
                    print("CAUTION: you requested fatigue load processing but I have no DLC to treat for that! Turning off fatigue load processing.")
                    withDEL = False

                if withDEL:

                    # Indices where to find DELs for the various nodes:
                    colnames = DELs.columns
                    i_AB1Fn = np.zeros(nx,int)
                    i_AB1Ft = np.zeros(nx,int)
                    i_B1MLx = np.zeros(nx,int)
                    i_B1MLy = np.zeros(nx,int)
                    i_B1FLz = np.zeros(nx,int)
                    i_B1StU = np.zeros(nx,int)
                    i_B1StL = np.zeros(nx,int)
                    i_B1StTE = np.zeros(nx,int)
                    for i in range(nx):
                        # i_AB1Fn[i] = colnames.get_loc("AB1N%03iFn"%(i+1)) #local chordwise
                        # i_AB1Ft[i] = colnames.get_loc("AB1N%03iFt"%(i+1)) #local normal
                        i_AB1Fn[i] = colnames.get_loc("AB1N%03iFx"%(i+1)) #rotor normal
                        i_AB1Ft[i] = colnames.get_loc("AB1N%03iFy"%(i+1)) #rotor tangential
                        i_B1MLx[i] = colnames.get_loc("B1N%03iMLx"%(i+1))
                        i_B1MLy[i] = colnames.get_loc("B1N%03iMLy"%(i+1))
                        i_B1FLz[i] = colnames.get_loc("B1N%03iFLz"%(i+1))
                        i_B1StU[i] = colnames.get_loc(f"BladeSparU_Strain_Stn{i+1}")
                        i_B1StL[i] = colnames.get_loc(f"BladeSparL_Strain_Stn{i+1}")
                        i_B1StTE[i] = colnames.get_loc(f"BladeTE_Strain_Stn{i+1}")


                    for dlc_num in DLCs_fat: #TODO: currently, if there are several dlc_num, DEL_life_B1 gets overwritten below
                        dlc = DLCs_fat[dlc_num]

                        nSEEDdel = dlc['nsims'] / len(dlc['U'])
                        # nVELdel = len(dlc["wind_speed"])

                        #duration of  time series
                        Tj = dlc["Tsim"]

                        # ----------------------------------------------------------------------------------------------
                        #    probability of the turbine to operate in specific conditions. 
                        #XXX THIS IS NOW ALSO AVAILABLE IN DLCs, but maybe only for dlc1.2? Let's just recompute it.
                        pj = pp.prob_WindDist( dlc['U'], disttype='pdf')
                        pj = pj / np.sum(pj) #renormalizing so that the sum of all the velocity we simulated covers the entire life of the turbine
                        #--
                        # pj = np.ones(Nj) / Nj   #uniform probability instead

                        print("Weight of the series (probability):")
                        print(pj)
                        

                        # ----------------------------------------------------------------------------------------------
                        # -- Compute extrapolated lifetime DEL for life --  
                        
                        if not dlc['idx']:
                            print("Warning: I did not find required data among time series to compute DEL! They will end up being 0.")
                        else:
                            print(f"Time series {dlc['idx']} are being processed for DEL...")

                        # a. Obtain the equivalent number of cycles
                        #TODO: better handle this for various DLCs with different nvels and nseeds. We currently assume the same Tj and nSeed across all fatigue DLCs.
                        fj = Tlife / Tj * pj
                        n_life_eq = np.sum(fj * Tj * f_eq)

                        # b. Aggregate DEL
                        k=0
                        for ids in [i_AB1Fn,i_AB1Ft,i_B1MLx,i_B1MLy,i_B1FLz,i_B1StU,i_B1StL,i_B1StTE]:
                            #loop over the DELs from all time series and sum
                            for i in dlc['idx']: 
                                
                                ivel = dlc['U'].index(  float(DLCs[i]['URef']) )
                                #average the DEL over the seeds: just as if we aggregated all the seeds
                                DEL_life_B1[:,k] += fj[ivel] * npDelstar[i][ids] / nSEEDdel

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
                        #       So we have x upwards and y chordwise positive towards TE (see FASTv8 manual). This is
                        #       confirmed by the results that give positive M in x AND y, with moment in y (flapwise) larger and >0.
                        #       Note: this is NOT in principal elastic axes, just airfoil-aligned axes.
                        #    - WISDEM strain computation needs the moments in airfoil axes: x positive towards suction side, 
                        #       y positive towards TE. Then the strain module processes the input by swapping x and y and by rotating to the 
                        #       principal axes: the 1st principal direction is chordwise positive towards TE, and 2 is positive upwards. 
                        #       This is confirmed by the stifness properties: EI11 is lower than EI22 (lower stifness edgewise than flapwise). 
                        #       
                        #    CONCLUSION: ED frame and AIRFOIL frame are aligned, NO NEED TO CHANGE ANYTHING.
                        
                        # DEL_life_B1[:,2] = DEL_life_B1[:,2]
                        # DEL_life_B1[:,3] = DEL_life_B1[:,3]  
                        # DEL_life_B1[:,4] = DEL_life_B1[:,4]
                        

                        # d. Obtain the equivalent load corresponding to the equivalent strain in spars
                        #    We can use the factors y/EI11, x/EI22, 1/EA, already computed for the combili factors. 

                        fact_LtU = np.zeros(n_span)
                        fact_LtL = np.zeros(n_span)

                        ooEA = np.zeros(n_span)
                        yoEIxxU = np.zeros(n_span)
                        xoEIyyU = np.zeros(n_span)
                        yoEIxxL = np.zeros(n_span)
                        xoEIyyL = np.zeros(n_span)
                        yoEIxxTE = np.zeros(n_span)
                        xoEIyyTE = np.zeros(n_span)
                        
                        for i in range(n_span):
                            ooEA[i]  = combili_channels["BladeSparU_Strain_Stn%d"%(i+1) ]["B1N0%02dFLz"%(i+1)]
                            yoEIxxU[i] = combili_channels["BladeSparU_Strain_Stn%d"%(i+1) ]["B1N0%02dMLx"%(i+1)]
                            xoEIyyU[i] = combili_channels["BladeSparU_Strain_Stn%d"%(i+1) ]["B1N0%02dMLy"%(i+1)]
                            yoEIxxL[i] = combili_channels["BladeSparL_Strain_Stn%d"%(i+1) ]["B1N0%02dMLx"%(i+1)]
                            xoEIyyL[i] = combili_channels["BladeSparL_Strain_Stn%d"%(i+1) ]["B1N0%02dMLy"%(i+1)]
                            yoEIxxTE[i] = combili_channels["BladeTE_Strain_Stn%d"%(i+1) ]["B1N0%02dMLx"%(i+1)]
                            xoEIyyTE[i] = combili_channels["BladeTE_Strain_Stn%d"%(i+1) ]["B1N0%02dMLy"%(i+1)]

                            for e in combili_channels["BladeSparU_Strain_Stn%d"%(i+1) ].values():
                                fact_LtU[i] += e
                            for e in combili_channels["BladeSparL_Strain_Stn%d"%(i+1) ].values():
                                fact_LtL[i] += e


                        # ############# NOTE: DEBUGGING ##########################
                        # # # or recompute them... may need to turn on WISDEM dry run above 
                        # EA = wt_opt['rotorse.EA']
                        # EI11 = wt_opt['rotorse.rs.frame.EI11']
                        # EI22 = wt_opt['rotorse.rs.frame.EI22']
                        # yU = wt_opt['rotorse.xu_spar'] #swapping coordinates before rotating: Airfoil to Hansen frame
                        # xU = wt_opt['rotorse.yu_spar']
                        # yL = wt_opt['rotorse.xl_spar']
                        # xL = wt_opt['rotorse.yl_spar']
                        # yTE = wt_opt['rotorse.xu_te']
                        # xTE = wt_opt['rotorse.yu_te']
                        # alpha = wt_opt['rotorse.rs.frame.alpha']

                        # x1U = np.zeros(len(xU))
                        # y1U = np.zeros(len(yU))
                        # x1L = np.zeros(len(xU))
                        # y1L = np.zeros(len(yU))
                        # x1TE = np.zeros(len(xU))
                        # y1TE = np.zeros(len(yU))

                        # fxU = np.zeros(len(yU))
                        # fyU = np.zeros(len(yU))
                        
                        # for i in range(len(xU)):
                        #     #rotate the coordinates from 'swapped' airfoil frame to principal axes
                        #     ca = np.cos(np.deg2rad(alpha[i]))
                        #     sa = np.sin(np.deg2rad(alpha[i]))

                        #     x1U[i] =  xU[i] * ca + yU[i] * sa
                        #     y1U[i] = -xU[i] * sa + yU[i] * ca
                        #     x1L[i] =  xL[i] * ca + yL[i] * sa
                        #     y1L[i] = -xL[i] * sa + yL[i] * ca
                        #     x1TE[i] =  xTE[i] * ca + yTE[i] * sa
                        #     y1TE[i] = -xTE[i] * sa + yTE[i] * ca

                        #     fxU[i] =  -(sa * y1U[i] / EI11[i] - ca * x1U[i] / EI22[i]) #minus because left-handed to right-handed: positive strain must give traction
                        #     fyU[i] =  -(ca * y1U[i] / EI11[i] + sa * x1U[i] / EI22[i])
                            


                        # #-----------------
                        # plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
                    
                        # plt.plot(ooEA ,label="1/EA")
                        # plt.plot(yoEIxxU,label="*MLx")
                        # plt.plot(xoEIyyU,label="*MLy")
                        # plt.plot(1/EA *1.e3,'--',label="1/EA")
                        # plt.plot(fxU *1.e3,'--',label="*MLx")
                        # plt.plot(fyU *1.e3,'--',label="*MLy")
                        # # plt.plot(fact_LtU,'k--',label="sum")
                        # plt.ylabel("u")
                        # plt.xlabel("stn")
                        # plt.xlim([0,10])
                        # plt.ylim([-1e-7,1e-7])
                        # plt.legend()
                        # plt.savefig("combiliF_U.png")

                        # plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
                    
                        # plt.plot(ooEA ,label="1/EA")
                        # plt.plot(yoEIxxL,label="*MLx")
                        # plt.plot(xoEIyyL,label="*MLy")
                        # # plt.plot(fact_LtL,'k--',label="sum")
                        # plt.ylabel("l")
                        # plt.xlabel("stn")
                        # plt.xlim([0,10])
                        # plt.ylim([-1e-7,1e-7])
                        # plt.legend()
                        # plt.savefig("combiliF_L.png")

                        # # plt.show()
                        # #-----------------
                        ################

                        # # d.(option1)
                        # # Find the equivalent Mx,My,Fz that will give the same strain as the Damage-equivalent Life strain,
                        # #  and that also have the same ratios as DEMx,DEMy,DEFz (that is, the damage-eq loads not based on strain)
                        # Ltilde_life_B1[:,2] = DEL_life_B1[:,5] / (DEL_life_B1[:,2]/DEL_life_B1[:,4] * yoEIxxU + DEL_life_B1[:,3]/DEL_life_B1[:,4] * xoEIyyU + ooEA )
                        # Ltilde_life_B1[:,0] = DEL_life_B1[:,2]/DEL_life_B1[:,4] * Ltilde_life_B1[:,2]
                        # Ltilde_life_B1[:,1] = DEL_life_B1[:,3]/DEL_life_B1[:,4] * Ltilde_life_B1[:,2]

                        # Ltilde_life_B1[:,5] = DEL_life_B1[:,6] / (DEL_life_B1[:,2]/DEL_life_B1[:,4] * yoEIxxL + DEL_life_B1[:,3]/DEL_life_B1[:,4] * xoEIyyL + ooEA )
                        # Ltilde_life_B1[:,3] = DEL_life_B1[:,2]/DEL_life_B1[:,4] * Ltilde_life_B1[:,5]
                        # Ltilde_life_B1[:,4] = DEL_life_B1[:,3]/DEL_life_B1[:,4] * Ltilde_life_B1[:,5]

                        #--
                        # # d.(option2)
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
                        # # d.(option3)
                        # # Find the equivalent Mx,My,Fz that will give the same strain as the Damage-equivalent Life strain,
                        # #  and that simply have equal weights (i.e., Mx=My=Fz)
                        # Ltilde_life_B1[:,0] = DEL_life_B1[:,5] / (yoEIxxU + xoEIyyU + ooEA ) * 1.e3
                        # Ltilde_life_B1[:,1] = DEL_life_B1[:,5] / (yoEIxxU + xoEIyyU + ooEA ) * 1.e3
                        # Ltilde_life_B1[:,2] = DEL_life_B1[:,5] / (yoEIxxU + xoEIyyU + ooEA ) * 1.e3

                        # Ltilde_life_B1[:,3] = DEL_life_B1[:,6] / (yoEIxxL + xoEIyyL + ooEA ) * 1.e3
                        # Ltilde_life_B1[:,4] = DEL_life_B1[:,6] / (yoEIxxL + xoEIyyL + ooEA ) * 1.e3
                        # Ltilde_life_B1[:,5] = DEL_life_B1[:,6] / (yoEIxxL + xoEIyyL + ooEA ) * 1.e3

                        #--
                        # d.(option4)
                        # Find the equivalent Mx,My,Fz that will give the same strain as the Damage-equivalent Life strain,
                        #  for slightly different designs. In addition to the baseline, we should simulate slightly different 
                        #  beams and obtain the corresponding inertial/stiffness properties and DEstrain. We can then solve 
                        #  (potentially in the least square sense) for the tilde loads:
                        #
                        #     | y/EIxx1  -x/EIyy1  1/EA1 |   | Mx_tilde |   | DEstrain1 |
                        #     | y/EIxx2  -x/EIyy2  1/EA2 | * | My_tilde | = | DEstrain2 |
                        #     | y/EIxx3  -x/EIyy3  1/EA3 |   | Fz_tilde |   | DEstrain3 |
                        #     | ...      ...       ...   |                  | ...       |
                        # 
                        # XXX: This procedure should be done dynamically here by perturbing the current DVs 4 times, performing 4 additional 
                        #   OpenFAST simulations, and re-solving (least-square) again. Currently though,
                        #   we just do it once a priori on the baseline design, and we will use the same
                        #   tilde loads for the hifi optimization throughout. The only thing we do here is to rescale the
                        #   'a priori' tilde loads by the ratio between the current DE strain and the 'a priori' DE strain.

                        if os.path.isfile("DEL_Tilde_loads_LstSqr.yaml"):
                            Ltilde_LstSqrSol = load_yaml("DEL_Tilde_loads_LstSqr.yaml")

                            Ltilde_life_B1[:,0] = DEL_life_B1[:,5] * Ltilde_LstSqrSol["DEL_Tilde_ss"]["deMLxPerStrain"]
                            Ltilde_life_B1[:,1] = DEL_life_B1[:,5] * Ltilde_LstSqrSol["DEL_Tilde_ss"]["deMLyPerStrain"]
                            Ltilde_life_B1[:,2] = DEL_life_B1[:,5] * Ltilde_LstSqrSol["DEL_Tilde_ss"]["deFLzPerStrain"]

                            Ltilde_life_B1[:,3] = DEL_life_B1[:,6] * Ltilde_LstSqrSol["DEL_Tilde_ps"]["deMLxPerStrain"]
                            Ltilde_life_B1[:,4] = DEL_life_B1[:,6] * Ltilde_LstSqrSol["DEL_Tilde_ps"]["deMLyPerStrain"]
                            Ltilde_life_B1[:,5] = DEL_life_B1[:,6] * Ltilde_LstSqrSol["DEL_Tilde_ps"]["deFLzPerStrain"]
                        else:
                            print("WARNING: I did not find 'DEL_Tilde_loads_LstSqr.yaml' so I can''t go for opt4. Reverting to Opt. 3.")
                            Ltilde_life_B1[:,0] = DEL_life_B1[:,5] / (yoEIxxU + xoEIyyU + ooEA ) * 1.e3
                            Ltilde_life_B1[:,1] = DEL_life_B1[:,5] / (yoEIxxU + xoEIyyU + ooEA ) * 1.e3
                            Ltilde_life_B1[:,2] = DEL_life_B1[:,5] / (yoEIxxU + xoEIyyU + ooEA ) * 1.e3

                            Ltilde_life_B1[:,3] = DEL_life_B1[:,6] / (yoEIxxL + xoEIyyL + ooEA ) * 1.e3
                            Ltilde_life_B1[:,4] = DEL_life_B1[:,6] / (yoEIxxL + xoEIyyL + ooEA ) * 1.e3
                            Ltilde_life_B1[:,5] = DEL_life_B1[:,6] / (yoEIxxL + xoEIyyL + ooEA ) * 1.e3

                        #--
                        print("Damage eq loads:")
                        print(np.transpose(DEL_life_B1))


                # ----------------------------------------------------------------------------------------------
                # -- proceed to extreme load/gust extrapolation if requested

                if withEXTR and not DLCs_extr:
                    print("CAUTION: you requested exrtreme load processing but I have no DLC to treat for that! Turning off extreme load processing.")
                    withEXTR = False

                if withEXTR:

                    # common values
                    dt = modeling_options["Level3"]["simulation"]["DT"]
                    
                    # Computing modulation of the range
                    locs = np.linspace(0.,1.,nx)
                    itrp_kern = scItrp.interp1d( rng_modulation_x, rng_modulation_val, )
                    rng_mod = itrp_kern(locs)                         

                    for dlc_num in DLCs_extr: 
                        dlc = DLCs_extr[dlc_num]

                        nSEEDextr = dlc['nsims'] / len(dlc['U'])

                        #duration of  time series
                        Tj = dlc["Tsim"]
                        nt = int( Tj / dt ) + 1 #not used anymore

                        print(iec_dlc_meth)
                        extr_meth = iec_dlc_meth[dlc_num]

                        dlc["extr_loads"] = []
                        dlc["extr_params"] = []
                        dlc["extr_loads_LR"] = []
                        
                        
                        # ----------------------------------------------------------------------------------------------
                        #    probability of the turbine to operate in specific conditions. 
                        pj_extr = pp.prob_WindDist(dlc['U'], disttype='pdf')
                        pj_extr = pj_extr / np.sum(pj_extr) #renormalizing so that the sum of all the velocity we simulated covers the entire life of the turbine

                        
                        if not dlc['idx']:
                            print("Warning: I did not find required data among time series to compute extreme loads! They will end up being 0.")
                        else:
                            print(f"Time series {dlc['idx']} are being processed for extreme loads...")

                        # Init our extreme loads
                        if dontAggregateExtreme:
                            n_aggr = dlc['nsims']
                        else:
                            n_aggr = 1
                        EXTR_distro_B1 = np.zeros([nx,n_processed,nbins,n_aggr])    
                        # if extremeExtrapMeth ==2: #DEPREC
                        #     EXTR_data_B1 = np.zeros([nx,5,nt])    

                        # Bin the load distribution of all timeseries, and potentially aggregate them
                        jloc = -1
                        for i in dlc['idx']:
                                jloc += 1        

                                # Because raw data are not available as such, need to go back look into the output files:
                                fname = simfolder + os.sep + fast_fnames[i]
                                print(fname)
                                fulldata = myOpenFASTread(fname, addExt=modeling_options["Level3"]["simulation"]["OutFileFmt"], combili_channels=combili_channels)

                                ivel = dlc['U'].index(  float(DLCs[i]['URef']) ) #index in the velocity probability distribution, different from the sim index if there are several seeds
                            
                                k = 0
                                for lab in ["AB1N%03iFx","AB1N%03iFy","B1N%03iMLx","B1N%03iMLy","B1N%03iFLz","BladeSparU_Strain_Stn%i","BladeSparL_Strain_Stn%i","BladeTE_Strain_Stn%i"]:
                                    # print(f"[{lab}] v{ivel} s{iseed} loc{i} - {fast_fnames[i]} {fast_dlclist[jloc]}")
                                    for ii in range(nx):
                                        hist, bns = np.histogram(fulldata[lab%(ii+1)], bins=nbins, range=rng[k])

                                        if dontAggregateExtreme:
                                            EXTR_distro_B1[ii,k,:,jloc] =  hist
                                        else:
                                            #average the EXTRM over the seeds: just as if we aggregated all the seeds
                                            EXTR_distro_B1[ii,k,:,0] +=  hist * pj_extr[ivel] / nSEEDextr

                                        # if extremeExtrapMeth ==2:
                                        #     EXTR_data_B1[i,j,:] = fulldata[lab%(i+1)]

                                    k+=1

                                del(fulldata)

                        #normalizing the distributions
                        for k in range(n_processed):
                            # dx = (rng[k][1]-rng[k][0])/(nbins)
                            # normFac = 1 / (nt * dx) #normalizing factor, to bring the EXTR_distro count into non-dimensional proability
                            # EXTR_distro_B1[:,k,:] *= normFac 
                            #--> there might be missing timesteps or anything... let's just make sure the distro sum to 1.00
                            for i in range(nx):
                                dx = (rng[k][1]-rng[k][0])/(nbins) * rng_mod[k,i]
                                for j in range(n_aggr):
                                    EXTR_distro_B1[i,k,:,j] /= np.sum(EXTR_distro_B1[i,k,:,j]*dx)

                        dlc["binned_loads"] = EXTR_distro_B1



                        IEC_50yr_prob = 1. - dt / Textr #=return period 50yr
                        # Explanation: This is only due to how we fit the probability distro
                        #   - I normalize the histogtam by normFac, i.e. the Y axis is 1/load. The integral of the histogram gives a density of 1.0 (unitless)
                        #   - Thus, instead of computing prob with T/50yr, I do dt/50yr since I normalized already with nt.
                        #   - There is no time in this density function, except implicitely the sampling period. 
                        #   - There is no time in this density function, except implicitely the sampling period. 
                        #   - There is no time in this density function, except implicitely the sampling period. 
                        #   - The max load that cooresponds to a return period of 2dT has a probability of 1-dt/2dt = 0.5, that is the mean of our distribution.
                        #   - So the prob should read: 50yr corresponds to 
                        #   - So the prob should read: 50yr corresponds to 
                        #   - So the prob should read: 50yr corresponds to 
                        #   - What's conterintuitive to me: the smaller the dt, the larger the 50yr. load... but this might be ok: if dt increases, you have a better certainty on the distro.

                        EXTR_life_LR = []
                        for j in range(n_aggr):
                            if extr_meth ==0:
                                # EXTR_life_B1, EXTR_distr_p, sice = exut.determine_max(rng, EXTR_distro_B1[:,:,:,j])
                                EXTR_life_B1, EXTR_distr_p = exut.extrapolate_extremeLoads_curveFit(rng, EXTR_distro_B1[:,:,:,j], ["maxForced",]*n_processed, IEC_50yr_prob, truncThr=truncThr, logfit=logfit, killUnder=killUnder, rng_mod=rng_mod)
                                raise ValueError("DEPRECATED EXTREME LOAD EXTRAPOLATION METHOD.")
                            elif extr_meth ==1:
                                #assumes only normal
                                EXTR_life_B1, EXTR_distr_p, side = exut.extrapolate_extremeLoads_hist(rng, EXTR_distro_B1[:,:,:,j],IEC_50yr_prob) #TODO: rng_mod
                                raise ValueError("DEPRECATED EXTREME LOAD EXTRAPOLATION METHOD.")
                            # elif extremeExtrapMeth ==2: #DEPREC
                            #     EXTR_life_B1, EXTR_distr_p = exut.extrapolate_extremeLoads(EXTR_data_B1[:,:,:,j], distr, IEC_50yr_prob)
                            elif extr_meth ==3:
                                EXTR_life_B1, EXTR_distr_p, side, EXTR_life_LR = exut.extrapolate_extremeLoads_curveFit(rng, EXTR_distro_B1[:,:,:,j], distr, IEC_50yr_prob, truncThr=truncThr, logfit=logfit, killUnder=killUnder, rng_mod=rng_mod)

                            # save data to the dict.
                            dlc["extr_loads"].append(EXTR_life_B1)
                            dlc["extr_params"].append(EXTR_distr_p)
                            dlc["extr_loads_LR"].append(EXTR_life_LR)
                        

                    # ------------ TIMIMGS ------------    
                    wt = mytime()
                    elapsed_postpro = wt - wt_postpro

                    # ------------ DUMPING RESULTS ------------    
                    if saveExtrNpy:
                        np.savez(saveExtrNpy, rng=rng, rng_mod=rng_mod, nbins=nbins, DLCs_extr=DLCs_extr, distr=distr, truncThr=truncThr, dt=dt, side=side)
                
                    # ------------ PLOTTING ------------    
                    for k in range(n_processed):
                        stp = (rng[k][1]-rng[k][0])/(nbins)
                        xbn_ = np.arange(rng[k][0]+stp/2.,rng[k][1],stp) #(bns[:-1] + bns[1:])/2.
                        dx = (rng[k][1]-rng[k][0])/(nbins)
                        
                        f1,ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
                        ax1.set_xlabel(labs[k])

                        f2,ax2 = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
                        ax2.set_yscale('log')
                        ax2.set_xlabel(labs[k])
                        ax2.set_ylim([ (1.-IEC_50yr_prob)/2. , 2.])         

                        xx_ = np.arange(rng[k][0]+dx/2.,rng[k][1],dx)       
  
                        for dlc_num in DLCs_extr: 
                            dlc = DLCs_extr[dlc_num] 

                            j = 0 #just plot the first timeseries or the aggregated one

                            for i in [5,15,25]: #arbiratry number spanwise stations
                                xx=xx_*rng_mod[k,i]
                                xbn=xbn_*rng_mod[k,i]

                                ss1 = ax1.plot(xbn,dlc["binned_loads"][i,k,:,j] )
                                c1 = ss1[0].get_color()
                                ax1.plot(dlc["extr_loads"][j][i,k] , 0, 'x' , color=c1)
                            
                                if side[k,i]>0: 
                                    #right extrap
                                    dsf1= 1.-np.cumsum(dlc["binned_loads"][i,k,:,j] )*dx 
                                    cpl = 0. #survaval function.
                                    cplf = 1. #survival function. 
                                else:
                                    #left extrap
                                    dsf1= np.cumsum(dlc["binned_loads"][i,k,:,j] )*dx 
                                    cpl = 1. #cdf function
                                    cplf = -1. #cdf function

                                dsf1[(dsf1>=1.-1E-16) | (dsf1<=1E-16)] = np.nan
                                ax2.plot(xbn,dsf1)
                                ax2.plot(dlc["extr_loads"][j][i,k] , 1.-IEC_50yr_prob, 'x' , color=c1)
                            
                            
                                if extr_meth > 0:

                                    print(dlc["extr_params"][j][i,k,:])

                                    if "twiceMaxForced" in distr[k]:
                                        pass
                                    elif "normForced" in distr[k]:
                                        ax1.plot(xx, stats.norm.pdf(xx, loc = dlc["extr_params"][j][i,k,0], scale = dlc["extr_params"][j][i,k,1]),'--', alpha=0.6 , color=c1)
                                        ax2.plot(xx, cpl+cplf*stats.norm.sf(xx, loc = dlc["extr_params"][j][i,k,0], scale = dlc["extr_params"][j][i,k,1]),'--', alpha=0.6 , color=c1)
                                    elif "norm" in distr[k] or "gumbel" in distr[k]: #2params models
                                        this = getattr(stats,distr[k])
                                        ax1.plot(xx, this.pdf(xx, loc = dlc["extr_params"][j][i,k,0], scale = dlc["extr_params"][j][i,k,1]),'--', alpha=0.6 , color=c1)
                                        ax2.plot(xx, cpl+cplf*this.sf(xx, loc = dlc["extr_params"][j][i,k,0], scale = dlc["extr_params"][j][i,k,1]),'--', alpha=0.6 , color=c1)
                                    else: #3params models
                                        this = getattr(stats,distr[k])
                                        ax1.plot(xx, this.pdf(xx, dlc["extr_params"][j][i,k,0], loc = dlc["extr_params"][j][i,k,1], scale = dlc["extr_params"][j][i,k,2]),'--', alpha=0.6 , color=c1)
                                        ax2.plot(xx, cpl+cplf*this.sf(xx, dlc["extr_params"][j][i,k,0], loc = dlc["extr_params"][j][i,k,1], scale = dlc["extr_params"][j][i,k,2]),'--', alpha=0.6 , color=c1)
                        f1.savefig(f"fit_{labs[k].split(' ')[0]}_{distr[k]}.png")
                        f2.savefig(f"fit_sf_{labs[k].split(' ')[0]}_{distr[k]}.png")

                        #NOTE: CAUTION: the sign on these plots correspond to ELASTODYN convention
                        # z: along the blade
                        # x: positive streamwise
                        # y: complete the triad, positive towards TE (left)
                    if showPlots:
                        plt.show()



                    # ------------ MORE PROCESSING ------------
                  
                    for dlc_num in DLCs_extr: 
                        dlc = DLCs_extr[dlc_num] 
                        for j in range(len(dlc["extr_loads"])):
                            # switch from IEC local blade frame to "AIRFOIL frame" with y positive towards TE
                            # dlc["extr_loads"][j][:,3] = -dlc["extr_loads"][j][:,3] #NOPE! the sign of y axis in ED and AIRFOIL frames is consistent

                            for k in range(n_processed):
                                    dlc["extr_loads"][j][:,k] *= fac[k]
                                    dlc["extr_loads_LR"][j][:,k,:] *= fac[k]


                # ----------------------------------------------------------------------------------------------
                # -- DUMPING THE RESULTS TO A FILE

                if dontAggregateExtreme:
                    #TODO: merge the following with DEL stuff below

                    # ----------------------------------------------------------------------------------------------
                    # -- write the analysis file
                    schema = load_yaml(fname_analysis_options)
                    #could use load_analysis_yaml from weis instead

                    locs = np.linspace(0.,1.,nx) #XXX Is this right? Shoule we obtain it from rotorse instead?


                    if withEXTR:
                        schema["extreme"] = {}
                        schema["extreme"]["grid_nd"] = locs.tolist()

                        for dlc_num in DLCs_extr: 
                            schema["extreme"][dlc_num] = {}
                            schema["extreme"][dlc_num]["U"] = dlc["U"]
                            schema["extreme"][dlc_num]["nsims"] = dlc["nsims"]
                            schema["extreme"][dlc_num]["MLx"] = []
                            schema["extreme"][dlc_num]["MLy"] = []
                            schema["extreme"][dlc_num]["FLz"] = []

                            schema["extreme"][dlc_num]["MLx_avg"] = []
                            schema["extreme"][dlc_num]["MLy_avg"] = []
                            schema["extreme"][dlc_num]["FLz_avg"] = []

                            schema["extreme"][dlc_num]["MLx_std"] = []
                            schema["extreme"][dlc_num]["MLy_std"] = []
                            schema["extreme"][dlc_num]["FLz_std"] = []

                            #filling: we assume that the param vector has [average, max or std, xxx]
                            for j in range( dlc["nsims"] ):
                                schema["extreme"][dlc_num]["MLx"].append( dlc["extr_loads"][j][:,2].tolist() )
                                schema["extreme"][dlc_num]["MLy"].append( dlc["extr_loads"][j][:,3].tolist() )
                                schema["extreme"][dlc_num]["FLz"].append( dlc["extr_loads"][j][:,4].tolist() )

                                #XXX CAUTION: should multiply by fac here!! finally done in the postpro script but still
                                schema["extreme"][dlc_num]["MLx_avg"].append( dlc["extr_params"][j][:,2,0].tolist() )
                                schema["extreme"][dlc_num]["MLy_avg"].append( dlc["extr_params"][j][:,3,0].tolist() )
                                schema["extreme"][dlc_num]["FLz_avg"].append( dlc["extr_params"][j][:,4,0].tolist() )

                                schema["extreme"][dlc_num]["MLx_std"].append( dlc["extr_params"][j][:,2,1].tolist() )
                                schema["extreme"][dlc_num]["MLy_std"].append( dlc["extr_params"][j][:,3,1].tolist() )
                                schema["extreme"][dlc_num]["FLz_std"].append( dlc["extr_params"][j][:,4,1].tolist() )

                    fname_analysis_options_struct = mydir + os.sep + "analysis_options_struct_withUnsteadyLoads.yaml"
                    my_write_yaml(schema, fname_analysis_options_struct)
                    #could use write_analysis_yaml from weis instead

                else:
                    # ----------------------------------------------------------------------------------------------
                    # -- Create a descriptor:

                    #Just a string written in the DEL export files to describe what's in there
                    if DLCs_fat and len(DLCs_fat)>0:
                        del_descr_str = "DEL computed based on DLC %s with %s seeds and the following vels: %s"%(
                            list(DLCs_fat), [DLCs_fat[k]['nsims']/len(DLCs_fat[k]['U']) for k in DLCs_fat], [DLCs_fat[k]['U'] for k in DLCs_fat]
                        )
                    else:
                        del_descr_str = "DEL unavailable"

                    if DLCs_extr and len(DLCs_extr)>0:
                        extr_descr_str = "extreme loading computed based on DLC %s with %s seeds and the following vels: %s"%(
                            list(DLCs_extr), [DLCs_extr[k]['nsims']/len(DLCs_extr[k]['U']) for k in DLCs_extr], [DLCs_extr[k]['U'] for k in DLCs_extr]
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
                        schema["DEL"]["StrainSparU"] = DEL_life_B1[:,5].tolist()
                        schema["DEL"]["StrainSparL"] = DEL_life_B1[:,6].tolist()
                        schema["DEL"]["StrainTE"] = DEL_life_B1[:,7].tolist()

                        schema["DEL_Tilde_ss"] = {}
                        schema["DEL_Tilde_ss"]["deMLx"] = Ltilde_life_B1[:,0].tolist()
                        schema["DEL_Tilde_ss"]["deMLy"] = Ltilde_life_B1[:,1].tolist()
                        schema["DEL_Tilde_ss"]["deFLz"] = Ltilde_life_B1[:,2].tolist()
                        schema["DEL_Tilde_ps"] = {}
                        schema["DEL_Tilde_ps"]["deMLx"] = Ltilde_life_B1[:,3].tolist()
                        schema["DEL_Tilde_ps"]["deMLy"] = Ltilde_life_B1[:,4].tolist()
                        schema["DEL_Tilde_ps"]["deFLz"] = Ltilde_life_B1[:,5].tolist()


                    if withEXTR:
                        schema["extreme"] = {}
                        schema["extreme"]["description"] = extr_descr_str
                        schema["extreme"]["grid_nd"] = locs.tolist()
                        schema["extreme"]["deMLx"] = dlc["extr_loads"][0][:,2].tolist()
                        schema["extreme"]["deMLy"] = dlc["extr_loads"][0][:,3].tolist()
                        schema["extreme"]["deFLz"] = dlc["extr_loads"][0][:,4].tolist()
                        schema["extreme"]["StrainSparL"] = dlc["extr_loads"][0][:,5].tolist()
                        schema["extreme"]["StrainSparU"] = dlc["extr_loads"][0][:,6].tolist()
                        schema["extreme"]["StrainTE"] = dlc["extr_loads"][0][:,7].tolist()

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

                    # EXPORT FOR WISDEM
                    fname_analysis_options_struct = mydir + os.sep + "analysis_options_struct_withDEL.yaml"
                    my_write_yaml(schema, fname_analysis_options_struct)
                    #could use write_analysis_yaml from weis instead

                    # EXPORT FOR MACH
                    if withDEL:
                        fname_fatMach = mydir + os.sep + "fatigueLoads_forMACH.yaml"
                        fat_schema = {}
                        fat_schema["DEL_Tilde_ps"] = schema["DEL_Tilde_ps"]
                        fat_schema["DEL_Tilde_ss"] = schema["DEL_Tilde_ss"]
                        my_write_yaml(fat_schema, fname_fatMach)
                    if withEXTR:
                        fname_extrMach = mydir + os.sep + "extrLoads_forMACH.yaml"
                        extr_schema = {}
                        extr_schema["extreme"] = schema["extreme"]
                        my_write_yaml(extr_schema, fname_extrMach)

                    schema_hifi = {}
                    if withDEL:
                        schema_hifi["DEL"] = {}
                        schema_hifi["DEL"]["description"] = del_descr_str
                        schema_hifi["DEL"]["grid_nd"] = locs.tolist()
                        schema_hifi["DEL"]["mean"] = {}
                        schema_hifi["DEL"]["mean"]["Fn"] = DEL_life_B1[:,0].tolist()
                        schema_hifi["DEL"]["mean"]["Ft"] = DEL_life_B1[:,1].tolist()
                    if withEXTR:
                        schema_hifi["extreme"] = {}
                        schema_hifi["extreme"]["description"] = extr_descr_str
                        schema_hifi["extreme"]["grid_nd"] = locs.tolist()
                        schema_hifi["extreme"]["mean"] = {}
                        schema_hifi["extreme"]["mean"]["Fn"] = dlc["extr_loads"][0][:,0].tolist()
                        schema_hifi["extreme"]["mean"]["Ft"] = dlc["extr_loads"][0][:,1].tolist()

                    my_write_yaml(schema_hifi, fname_aggregatedEqLoads)


                # ----------------------------------------------------------------------------------------------
                # -- Final plots
                for k in range(n_processed):
                    plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
                    if withEXTR:
                        for dlc_num in DLCs_extr: 
                            dlc = DLCs_extr[dlc_num] 
                            for j in range(len(dlc["extr_loads"])):
                                a = plt.plot(locs,dlc["extr_loads"][j][:,k], label=f'{dlc_num}')
                                if len(dlc["extr_loads_LR"][j])>0:
                                    plt.plot(locs,dlc["extr_loads_LR"][j][:,k,0], '--', color=a[0].get_color())
                                    plt.plot(locs,dlc["extr_loads_LR"][j][:,k,1], '--', color=a[0].get_color())

                    if withDEL:
                        plt.plot(locs,DEL_life_B1[:,k] , label="DEL")
                    plt.ylabel(labs[k])
                    plt.xlabel("r/R")
                    plt.legend()
                    plt.savefig(f"{labs[k].split(' ')[0]}.png")

                if withDEL:
                    plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
                    for k in [0,1,2]:
                        plt.plot(locs,Ltilde_life_B1[:,k] , label=labs_Lt[k])
                        plt.ylabel("U")
                        plt.xlabel("r/R")
                        plt.legend()
                        plt.savefig("LtildeU.png")

                    plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
                    for k in [3,4,5]:
                        plt.plot(locs,Ltilde_life_B1[:,k] , label=labs_Lt[k])
                        plt.ylabel("L")
                        plt.xlabel("r/R")
                        plt.legend()
                        plt.savefig("LtildeL.png")
                    
                    # ############# NOTE: DEBUGGING ##########################
                    # #Verify the solving: the tilde stresses give the correct strains
                    # plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
                    # plt.plot(locs,(Ltilde_life_B1[:,0]*yoEIxxU+Ltilde_life_B1[:,1]*xoEIyyU+Ltilde_life_B1[:,2]*ooEA) *1e-3, label="U")
                    # plt.plot(locs,(Ltilde_life_B1[:,3]*yoEIxxL+Ltilde_life_B1[:,4]*xoEIyyL+Ltilde_life_B1[:,5]*ooEA) *1e-3, label="L")
                    # # plt.plot(locs,(Ltilde_life_B1[:,0]*yoEIxxTE+Ltilde_life_B1[:,1]*xoEIyyTE+Ltilde_life_B1[:,2]*ooEA ) *1e-3, label="TE")

                    # # Double check!!
                    # # ROTATE LOADS
                    # M1 = np.zeros(len(xU))
                    # M2 = np.zeros(len(yU))
                    
                    # for i in range(len(xU)):
                    #     #rotate the coordinates from 'swapped' airfoil frame to principal axes
                    #     ca = np.cos(np.deg2rad(alpha[i]))
                    #     sa = np.sin(np.deg2rad(alpha[i]))
                    #     M1[i] =  Ltilde_life_B1[i,1] * ca + Ltilde_life_B1[i,0] * sa
                    #     M2[i] = -Ltilde_life_B1[i,1] * sa + Ltilde_life_B1[i,0] * ca

                    # plt.plot(locs,(  -(M1*y1U/EI11 -M2*x1U/EI22 ) +Ltilde_life_B1[:,2]/EA) ,'--', label="U")


                    # for i in range(len(xU)):
                    #     #rotate the coordinates from 'swapped' airfoil frame to principal axes
                    #     ca = np.cos(np.deg2rad(alpha[i]))
                    #     sa = np.sin(np.deg2rad(alpha[i]))
                    #     M1[i] =  Ltilde_life_B1[i,4] * ca + Ltilde_life_B1[i,3] * sa
                    #     M2[i] = -Ltilde_life_B1[i,4] * sa + Ltilde_life_B1[i,3] * ca
                        
                    # plt.plot(locs,(  -(M1*y1L/EI11 -M2*x1L/EI22 ) +Ltilde_life_B1[:,5]/EA) ,'--', label="L")
                    # # plt.plot(locs,(  -(M1*y1TE/EI11-M2*x1TE/EI22) +Ltilde_life_B1[:,2]/EA) ,'--', label="TE")

                    # plt.xlabel("r/R")
                    # plt.legend()
                    # plt.savefig("strain2.png")
                    ############# 


                if showPlots:
                    plt.show()
        elif fname_analysis_options_FORCED:
            #if you already postprocessed the data above, and want to do the lofi optimization
            print(f"Forced use of analysis file: {fname_analysis_options_FORCED}\nI will not check that this file contains DEL or EXTRM info. Please make sure it matches your current request.")
            fname_analysis_options_struct = fname_analysis_options_FORCED
        else:
            fname_analysis_options_struct = mydir + os.sep + "analysis_options_struct.yaml"

        if MPI:
            MPI.COMM_WORLD.Barrier()

        if rank == 0:
            wt = mytime()
            wt_optim = wt

        # +++++++++++++++++++++++++++++++++++++++
        #           PHASE 2 : Optimize
        # +++++++++++++++++++++++++++++++++++++++
        if doLofiOptim:
            # Let's use the most up-to-date turbine as a starting point:
            wt_opt, analysis_options, opt_options = run_wisdem(current_wt_input, fname_modeling_options, fname_analysis_options_struct)

            print("\n\n\n  -------------- DONE WITH WISDEM ------------------\n\n\n\n")    

        if rank == 0:
            wt = mytime()
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
            if os.path.isfile(fname_extrMach):
                os.system(f"cp {fname_extrMach} {folder_arch + os.sep}")
            if os.path.isfile(fname_fatMach):
                os.system(f"cp {fname_fatMach} {folder_arch + os.sep}")
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
                os.system(f"mv {saveExtrNpy} {folder_arch + os.sep}")        
            if os.path.isfile(mydir + os.sep + modeling_options["WISDEM"]["RotorSE"]["exportRawPrecompFile"]):
                os.system(f'mv {modeling_options["WISDEM"]["RotorSE"]["exportRawPrecompFile"]} {folder_arch + os.sep + modeling_options["WISDEM"]["RotorSE"]["exportRawPrecompFile"]}')
            if os.path.isfile(mydir + os.sep + modeling_options["WISDEM"]["RotorSE"]["exportMaxCoordsFile"]):
                os.system(f'mv {modeling_options["WISDEM"]["RotorSE"]["exportMaxCoordsFile"]} {folder_arch + os.sep + modeling_options["WISDEM"]["RotorSE"]["exportMaxCoordsFile"]}')

            figdir = folder_arch + os.sep + 'figs' 
            if not os.path.isdir(figdir):
                os.makedirs(figdir)
            thisfdir = figdir + os.sep + currFolder
            if not os.path.isdir(thisfdir):
                os.makedirs(thisfdir)
            os.system(f"mv *.png {thisfdir}")

            # --- TIMINGS --
            wt = mytime()
            elapsed_tot = wt - wt_tot

            with open(folder_arch + os.sep + "timings.txt", "a") as file:
                file.write('%8.16E, %8.16E, %8.16E, %8.16E, %8.16E\n' % (wt, elapsed_tot, elapsed_sim, elapsed_postpro, elapsed_optim) )

        # update the path to the current optimal turbine
        current_wt_input = folder_arch + os.sep + "outputs_optim" + os.sep + currFolder + os.sep + "blade_out.yaml"

        #reset path to any precomputed data, so that if there is more iterations, we will actually do up-to-date computations
        readOutputFrom = "" 

    ## -- plot successive DEL --

    print(f"  ============== DONE AFTER {nGlobalIter} ITER ===================\n")



if __name__ == '__main__':
    # Running the script with default options

    ## File management
    mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file

    fname_wt_input = mydir + os.sep + "Madsen2019_composite_v02_IC.yaml"
    fname_modeling_options = mydir + os.sep + "modeling_options.yaml"
    fname_analysis_options = mydir + os.sep + "analysis_options_struct.yaml"
    
    XtrFat(
        #folder and files
        mydir, #folder where to find all the input files ()
        fname_wt_input, #input file
        fname_modeling_options, #input file
        fname_analysis_options, #input file
        showPlots = True,
    )