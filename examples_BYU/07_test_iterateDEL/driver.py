import os, sys
import yaml, copy

import numpy as np

from wisdem.commonse.mpi_tools import MPI
from XtrFat.XtrFat import XtrFat

# ---------------------
# Duplicate stdout to a file
# The following will not redirect OpenFAST output though.
class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()

if not MPI:
    # Let's make sure we log the output, because you are probably not running this via a batch script.
    sys.stdout = Tee('stdout.log','w')

# Should do something inspired by what they do here:
# https://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/
# of 
# https://python.tutorialink.com/capturing-print-output-from-shared-library-called-from-python-with-ctypes-module/


if __name__ == '__main__':

    #==================== DEFINITIONS  =====================================

    ## File management
    mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file

    fname_wt_input = mydir + os.sep + "Madsen2019_composite_v02_IC.yaml"

    fname_modeling_options = mydir + os.sep + "modeling_options.yaml"
    fname_analysis_options = mydir + os.sep + "analysis_options_struct.yaml"
    fname_analysis_options_WEIS = mydir + os.sep + "analysis_options_WEIS.yaml"
    fname_aggregatedEqLoads = mydir + os.sep + "aggregatedAeroLoads.yaml"

    folder_arch = mydir + os.sep + "results"


    withEXTR = True  #compute EXTREME moments 
    withDEL = True  #compute DEL moments - if both are False, the lofi optimization falls back to a default WISDEM run
    doLofiOptim = False  #skip lofi optimization, if you are only interested in getting the DEL and EXTR outputs (e.g. for HiFi)
    nGlobalIter = 1
    restartAt = 0

    readOutputFrom = "" #results path where to get output data. If not empty, we do bypass OpenFAST execution and only postprocess files in that folder instead
    # readOutputFrom = "results/sim/iter_0"
    # readOutputFrom = "Madsen2019_composite_v02_IC/sim/iter_0"

    #CAUTION: when specifying a readOutput, you must make sure that the modeling_option.yaml you provide actually correspond to those outputs (mostly the descrition of simulation time and IEC conditions)

    fname_analysis_options_FORCED = "" #if this analysis file is provides (with EXTREME and FATIGUE loads), the whole preprocessing is bypassed and we jump directly to the lofi optimization
    
    showPlots = not MPI #only show plots for non-mpi runs

    # +++++++++++ Design choice in EXTREME loads +++++++++++
    #-Binning-:
    #XXX: CAUTION: this required some manual tuning, and will need retuning for another turbine...

    # total number of bins
    nbins = 1000        
    
    # total range over which we bin, for each quantity monitored.
    #   The range should span the entire range of values we can get from the simultaion (from min to max)
    rng = [ (-2.e4,2.e4), #Fx [N/m]
            (-2.e4,2.e4),  #Fy [N/m]
            (-5.e4,5.e4),  #MLx [kNm]
            (-5.e4,5.e4),  #MLy [kNm]
            (-5.e3,5.e3),  #FLz [kN]
            (-6.e-3,6.e-3),  #StrainU [-]
            (-6.e-3,6.e-3),  #StrainL [-]
            (-6.e-3,6.e-3),  #StrainTE [-]
            ]
    # Since this range may be too large in some regions of the blades (e.g. at the tip, where the loads are smaller)
    #  we allow to modulate the range. rng will be scaled by a factor interpolated from rng_modulation based on
    #  the spanwise location.
    rng_modulation_x   = [0,.75,1]
    rng_modulation_val = [ [1,]*3 ,
                           [1,]*3 ,
                           [1,1,.25] ,
                           [1,1,.25] ,
                           [1,1,.25] ,
                           [1,1,.1] ,
                           [1,1,.1] ,
                           [1,1,.1] ,
                         ]


    #-Extreme load extrapolation-:
    extremeExtrapMeth = 3
    #0: just take the max of the observed loads during the timeseries
    #1: statistical moment-based method: just compute avg and std of the data, and rebuild a normal distribution for that
    #2: try the fit function of scipy.stats to the whole data: EXPERIMENTAL, and does not seem to be using it properly
    #3: curvefit the distributions to the histogramme - RECOMMENDED APPROACH
    logfit = True #True: fit the log of the survival function. False: fit the pdf
    killUnder = 1E-14 #remove all values in the experimental distribution under this threshold (numerical noise)

    #-- Assumed distr for each of the channels --
    # Note: 
    # - the longer the simulation window, the better (avoid at all cost to include the initial transient)
    # - the beam residual moments are well approximated by Gaussian
    # - the aerodynamic loads should better correspond to chi2 or weibull_min, however the fit is very sensitive to initial conditions
    # - use "normForced" as a distribution for a failsafe normal fitting (in case too many warning). It reverts back to moment-based fit. This will likely overestimate the extreme loads.
    # - because of the compounded gravitational and aero loads, MLx is bimodal... not very practival for a fit! :-(
    # - use "twiceMaxForced" as a distribution for a failsafe extreme load that amounts to twice the max recorded load.

    distr = ["norm","norm","norm","norm","norm","norm","norm","norm"]     # new recommended setup
    #NOTE:
    # [norm and 1.0] seems to be working well for bimodal distributions. That's the case for MLx,FLz,StrainTE.
    # Fn and Ft are skewed distributions, but their tail is actually well fitted by a normal. Ft could work with a gumbel_r
    # FLz has a super weird tri-modal shape
    # MLx is super symmetric wrt 0
    # MLy is super weird: the distribution is towards >0 but the tail is actually long towards the negative numbers, leading to an overall <0 extreme load...!
    #NOTE general:
    # weibull is a good distribution but it's not easy to use it for both left and right tails since it's skewed...
    # norm is definitely the easiest to use and gives the best fits anyway

    # -- Restrict the portion of data considered for the fit (keep the tail only) ---------
    truncThr = [0.5,1.0,1.0,0.5,1.0,0.5,0.5,1.0] 
                            
    saveExtrNpy = "extrmDistro.npz"
    dontAggregateExtreme = False #EXPERIMENTAL: consider each velocity,seed in each extreme DLC separately. When True, exports average and min/max load for each. Should have a saveExtrNpy.
       #XXX: currently, the aggregation in Extreme is done considering only DLC1.3. TODO: handle aggregation accross several DLCs?

    # +++++++++++ Design choice in fatigue: for how long do you size the turbine + other parameters +++++++++++
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


    XtrFat(
        #folder and files
        mydir, #folder where to find all the input files ()
        fname_wt_input, #input file
        fname_modeling_options, #input file
        fname_analysis_options, #input file
        fname_analysis_options_WEIS = fname_analysis_options_WEIS, #output file
        readOutputFrom = readOutputFrom,
        folder_arch=folder_arch,
        fname_aggregatedEqLoads=fname_aggregatedEqLoads, #output file
        #extreme:
        withEXTR = withEXTR,  #compute EXTREME moments 
        nbins = nbins, # total number of bins
        rng = rng,
        Textr = Textr,
        # extremeExtrapMeth = 3,
        truncThr = truncThr,
        distr = distr,
        # logfit = True,  
        # killUnder = 1E-14, 
        rng_modulation_x = rng_modulation_x, 
        rng_modulation_val = rng_modulation_val,
        # saveExtrNpy = "extrmDistro.npz",
        # dontAggregateExtreme = False,
        #fatigue:
        withDEL = withDEL,
        m_wohler = m_wohler,
        Tlife = Tlife,
        f_eq = f_eq,
        #driving:
        # doLofiOptim = False, #skip lofi optimization, if you are only interested in getting the DEL and EXTR outputs (e.g. for HiFi)
        # fname_analysis_options_FORCED="", #input file (to jump to lofi optim directly)
        # nGlobalIter = 1,
        # restartAt = 0,
        showPlots = showPlots,
    )


    
os.system(f"mv stdout.log {folder_arch + os.sep}")        
