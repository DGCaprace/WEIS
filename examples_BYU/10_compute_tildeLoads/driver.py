import os, sys
import yaml, copy

import numpy as np

from wisdem.commonse.mpi_tools import MPI
from XtrFat.XtrFat import *
# from XtrFat.processingTools import perturb_design

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



## File management
mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file

fname_wt_inputs = [
    "Madsen2019_composite_v02_IC.yaml",
    "Madsen2019_composite_v02_ICp01.yaml",
    "Madsen2019_composite_v02_ICp02.yaml",
    "Madsen2019_composite_v02_ICp03.yaml",
    "Madsen2019_composite_v02_ICp04.yaml",
]


Textr = 3600 * 24 * 365 * 10 #10 year return-period extreme event
Tlife = 3600 * 24 * 365 * 1 #1 year fatigue equivalent


#==================== COMMON DEFINITIONS FOR THE PROCESSING  =====================================
fname_modeling_options = "modeling_options_perturbation.yaml"
fname_analysis_options = "analysis_options_struct.yaml"
fname_aggregatedEqLoads = "aggregatedAeroLoads.yaml"


withEXTR = True  #compute EXTREME moments 
withDEL = True  #compute DEL moments - if both are False, the lofi optimization falls back to a default WISDEM run

readOutputFrom = "" #results path where to get output data. If not empty, we do bypass OpenFAST execution and only postprocess files in that folder instead

showPlots = not MPI #only show plots for non-mpi runs

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

#-Extreme load extrapolation-
extremeExtrapMeth = 3
logfit = True #True: fit the log of the survival function. False: fit the pdf
killUnder = 1E-14 #remove all values in the experimental distribution under this threshold (numerical noise)

distr = ["norm","norm","norm","norm","norm","norm","norm","norm"]     # new recommended setup
truncThr = [0.5,1.0,1.0,0.5,1.0,0.5,0.5,1.0] 
                        
saveExtrNpy = "extrmDistro.npz"
m_wohler = 10 #caution: also hardcoded in the definition of fatigue_channels at the top of runFAST_pywrapper 
f_eq = 1/Tlife #--> RECOMMENDED SETTING

readWindFrom = ''

for ip,fname in enumerate(fname_wt_inputs):

    fname_wt_input = mydir + os.sep + fname
    folder_arch = ".".join(fname_wt_input.split('.')[:-1]) #remove just the extension

    if ip>0:
        readWindFrom = os.path.join(mydir,".".join(fname_wt_inputs[0].split('.')[:-1]),'sim','iter_0','wind')

    XtrFat(
        #folder and files
        mydir, #folder where to find all the input files ()
        fname_wt_input, #input file
        fname_modeling_options, #input file
        fname_analysis_options, #input file
        # fname_analysis_options_WEIS = fname_analysis_options_WEIS, #output file
        readOutputFrom = readOutputFrom,
        readWindFrom = readWindFrom,
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


if not MPI:
    os.system(f"mv stdout.log {folder_arch + os.sep}")        
