import os
import yaml

from wisdem import run_wisdem
from wisdem.inputs import load_yaml, write_yaml #, validate_without_defaults, validate_with_defaults, simple_types
# from pCrunch import PowerProduction, LoadsAnalysis
# from pCrunch.io import OpenFASTAscii, OpenFASTBinary#, OpenFASTOutput

import sys, shutil
import numpy as np
import matplotlib.pyplot as plt


"""This script starts from a given turbine yaml and an external loading file, and just computes 1 evaluation in WISDEM.

:raises FileNotFoundError: [description]
:return: [description]
:rtype: [type]
"""

# ---------------------
def my_write_yaml(instance, foutput):
    if os.path.isfile(foutput):
        print(f"File {foutput} already exists... replacing it.")
        os.remove(foutput)
    # Write yaml with updated values
    with open(foutput, "w", encoding="utf-8") as f:
        yaml.dump(instance, f)



#==================== DEFINITIONS  =====================================

## File management
mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
fname_modeling_options = mydir + os.sep + "modeling_options.yaml"
fname_analysis_options = mydir + os.sep + "analysis_options_struct.yaml"

fname_wt_input = mydir + os.sep + "Madsen2019_10_forWEIS_isotropic.yaml"
fname_loads = mydir + os.sep + "Madsen2019_10_forWEIS_isotropic_ED/nominalLoads.yaml" #the loading condition you want to use (should have a nominal section)
folder_arch = mydir + os.sep + "LoFiEval_isotropic"

# fname_wt_input = mydir + os.sep + "Madsen2019_10_forWEIS.yaml"
# fname_loads = mydir + os.sep + "Madsen2019_10_forWEIS/nominalLoads.yaml" #the loading condition you want to use (should have a nominal section)
# folder_arch = mydir + os.sep + "LoFiEval_composite"


runWISDEM = True


#==================== ======== =====================================
## Preprocessing

withEXTR = True  #compute EXTREME moments, well more like the strin  based on the loading you gave as an input
withDEL = False  #leave this off: you don't need it 


# analysis_opt = load_yaml(fname_analysis_options)
# wt_init = load_yaml(fname_wt_input)
# modeling_options = load_yaml(fname_modeling_options)  #initial load


## Load the loading file
schema_loads = load_yaml(fname_loads)


## Update the analysis file
schema = load_yaml(fname_analysis_options)


schema["extreme"] = {}
schema["extreme"]["description"] = schema_loads["nominal"]["description"]
schema["extreme"]["grid_nd"] = schema_loads["nominal"]["grid_nd"]
schema["extreme"]["deMLx"] = schema_loads["nominal"]["deMLx"]
schema["extreme"]["deMLy"] = schema_loads["nominal"]["deMLy"]
schema["extreme"]["deFLz"] = schema_loads["nominal"]["deFLz"]
schema["constraints"]["blade"]["extreme_loads_from_user_inputs"] = True #we are using that channel as a way to specify a loading. The we will read the corresponding strain the EXTRM strain  output

schema["general"]["folder_output"] = folder_arch

fname_analysis_options_struct = mydir + os.sep + "analysis_options_struct_withMyLoading.yaml"
my_write_yaml(schema, fname_analysis_options_struct)


#==================== ======== =====================================
# Simulation

# overridden_values = {}
# overridden_values["rotorse.xu_strain_spar"] = [2.60001698,2.38170205,1.62793445,0.86687718,0.69488276,1.01063347
# ,1.26065757,1.14452006,0.97216279,0.85892484,0.76934834,0.72187777
# ,0.67424907,0.61707346,0.5643521,0.51985643,0.48173753,0.44862879
# ,0.41654838,0.38797813,0.36246771,0.33828892,0.31465849,0.29173286
# ,0.27105139,0.25037732,0.22966595,0.20409592,0.16973297,0.06727553]
# overridden_values["rotorse.xl_strain_spar"] = [-2.37821503,-2.31329415,-1.74729165,-1.02954568,-0.88685459,-1.25368234
# ,-1.50926078,-1.35840821,-1.15849323,-1.02089267,-0.91783497,-0.85902831
# ,-0.80425368,-0.73166315,-0.65537453,-0.59224189,-0.53825301,-0.49606488
# ,-0.46160527,-0.42963624,-0.39932503,-0.37087766,-0.34330666,-0.31686874
# ,-0.29312981,-0.26970749,-0.24675835,-0.21975102,-0.18601458,-0.07544804]
# overridden_values["rotorse.yu_strain_spar"] = [0.68800779,1.02697889,0.6198455,0.21196559,0.0473643,0.10595746
# ,0.13668742,0.02723982,-0.05129326,-0.08304079,-0.10412325,-0.10516968
# ,-0.10115421,-0.09713379,-0.08307498,-0.08361616,-0.0810186,-0.07931845
# ,-0.07438488,-0.07160432,-0.06875535,-0.06738904,-0.06542558,-0.06313923
# ,-0.05940534,-0.05573268,-0.05444918,-0.06011351,-0.09258093,-0.07271351]
# overridden_values["rotorse.yl_strain_spar"] = [-1.25674834,-1.35933239,-0.87715502,-0.51728909,-0.53327418,-0.7503961
# ,-0.91045496,-0.86330604,-0.75638271,-0.66742097,-0.59299581,-0.53817104
# ,-0.48307338,-0.42552989,-0.35949942,-0.31317682,-0.27125189,-0.23824789
# ,-0.20654436,-0.18094544,-0.15863941,-0.14068945,-0.12467393,-0.11076018
# ,-0.09766458,-0.08599894,-0.07803114,-0.07776103,-0.10558723,-0.07481666]

if runWISDEM:
    wt_opt, analysis_options, opt_options = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options_struct,
        # overridden_values = overridden_values
    )

    print("\n\n\n  -------------- DONE WITH WISDEM ------------------\n\n\n\n")    

#+x is ss, -x is ps
# may get +y on ss and -y on ps, but not always

#WHAT IT SHOULD BE 
# rotorse.xu_strain_spar,"[2.60001698 2.38170205 1.62793445 0.86687718 0.69488276 1.01063347
#  1.26065757 1.14452006 0.97216279 0.85892484 0.76934834 0.72187777
#  0.67424907 0.61707346 0.5643521  0.51985643 0.48173753 0.44862879
#  0.41654838 0.38797813 0.36246771 0.33828892 0.31465849 0.29173286
#  0.27105139 0.25037732 0.22966595 0.20409592 0.16973297 0.06727553]",x-position of midpoint of spar cap on upper surface for strain calculation
# rotorse.xl_strain_spar,"[-2.37821503 -2.31329415 -1.74729165 -1.02954568 -0.88685459 -1.25368234
#  -1.50926078 -1.35840821 -1.15849323 -1.02089267 -0.91783497 -0.85902831
#  -0.80425368 -0.73166315 -0.65537453 -0.59224189 -0.53825301 -0.49606488
#  -0.46160527 -0.42963624 -0.39932503 -0.37087766 -0.34330666 -0.31686874
#  -0.29312981 -0.26970749 -0.24675835 -0.21975102 -0.18601458 -0.07544804]",x-position of midpoint of spar cap on lower surface for strain calculation
# rotorse.yu_strain_spar,"[ 0.68800779  1.02697889  0.6198455   0.21196559  0.0473643   0.10595746
#   0.13668742  0.02723982 -0.05129326 -0.08304079 -0.10412325 -0.10516968
#  -0.10115421 -0.09713379 -0.08307498 -0.08361616 -0.0810186  -0.07931845
#  -0.07438488 -0.07160432 -0.06875535 -0.06738904 -0.06542558 -0.06313923
#  -0.05940534 -0.05573268 -0.05444918 -0.06011351 -0.09258093 -0.07271351]",y-position of midpoint of spar cap on upper surface for strain calculation
# rotorse.yl_strain_spar,"[-1.25674834 -1.35933239 -0.87715502 -0.51728909 -0.53327418 -0.7503961
#  -0.91045496 -0.86330604 -0.75638271 -0.66742097 -0.59299581 -0.53817104
#  -0.48307338 -0.42552989 -0.35949942 -0.31317682 -0.27125189 -0.23824789
#  -0.20654436 -0.18094544 -0.15863941 -0.14068945 -0.12467393 -0.11076018
#  -0.09766458 -0.08599894 -0.07803114 -0.07776103 -0.10558723 -0.07481666]"


#WHAT IT IS IN THE COMPOSITE CALL:
# rotorse.xu_strain_spar,"[-0.00237137 -0.02232077 -0.39312588 -0.59534197 -0.6264342  -0.44938689
#  -0.22481868 -0.26423804 -0.31571896 -0.39576025 -0.36414385 -0.32376401
#  -0.29368556 -0.27087079 -0.26695719 -0.24404249 -0.22878154 -0.20714033
#  -0.17887004 -0.15276627 -0.1293438  -0.10864371 -0.09048269 -0.07474432
#  -0.06135962 -0.04935499 -0.04080752 -0.03270521 -0.0227165  -0.00747272]",x-position of midpoint of spar cap on upper surface for strain calculation
# rotorse.xl_strain_spar,"[-0.00237137 -0.02232077 -0.39312588 -0.59534197 -0.6264342  -0.44938689
#  -0.22481868 -0.26423804 -0.31571896 -0.39576025 -0.36414385 -0.32376401
#  -0.29368556 -0.27087079 -0.26695719 -0.24404249 -0.22878154 -0.20714033
#  -0.17887004 -0.15276627 -0.1293438  -0.10864371 -0.09048269 -0.07474432
#  -0.06135962 -0.04935499 -0.04080752 -0.03270521 -0.0227165  -0.00747272]",x-position of midpoint of spar cap on lower surface for strain calculation
# rotorse.yu_strain_spar,"[-2.68994463 -2.6714235  -2.59877645 -2.59876147 -2.63808801 -2.55904653
#  -2.44879187 -2.46592522 -2.41897788 -2.3721729  -2.33443482 -2.26927189
#  -2.19455422 -2.12028532 -2.04105988 -1.95954551 -1.86181616 -1.76368786
#  -1.66177813 -1.56029523 -1.46038569 -1.36359875 -1.26859831 -1.17583691
#  -1.09181366 -1.00796706 -0.92424035 -0.86519809 -0.72733185 -0.2912122 ]",y-position of midpoint of spar cap on upper surface for strain calculation
# rotorse.yl_strain_spar,"[-2.68994463 -2.6714235  -2.59877645 -2.59876147 -2.63808801 -2.55904653
#  -2.44879187 -2.46592522 -2.41897788 -2.3721729  -2.33443482 -2.26927189
#  -2.19455422 -2.12028532 -2.04105988 -1.95954551 -1.86181616 -1.76368786
#  -1.66177813 -1.56029523 -1.46038569 -1.36359875 -1.26859831 -1.17583691
#  -1.09181366 -1.00796706 -0.92424035 -0.86519809 -0.72733185 -0.2912122 ]",y-position of midpoint of spar cap on lower surface for strain calculation

#WHAT IT IS IN THE ISOTROPIC CALL:
# rotorse.xu_strain_spar,"[-0.00237137 -0.02232077 -0.39312588 -0.59534197 -0.6264342  -0.44938689
#  -0.22481868 -0.26423804 -0.31571896 -0.39576025 -0.36414385 -0.32376401
#  -0.29368556 -0.27087079 -0.26695719 -0.24404249 -0.22878154 -0.20714033
#  -0.17887004 -0.15276627 -0.1293438  -0.10864371 -0.09048269 -0.07474432
#  -0.06135962 -0.04935499 -0.04080752 -0.03270521 -0.0227165  -0.00747272]",x-position of midpoint of spar cap on upper surface for strain calculation
# rotorse.xl_strain_spar,"[-0.00237137 -0.02232077 -0.39312588 -0.59534197 -0.6264342  -0.44938689
#  -0.22481868 -0.26423804 -0.31571896 -0.39576025 -0.36414385 -0.32376401
#  -0.29368556 -0.27087079 -0.26695719 -0.24404249 -0.22878154 -0.20714033
#  -0.17887004 -0.15276627 -0.1293438  -0.10864371 -0.09048269 -0.07474432
#  -0.06135962 -0.04935499 -0.04080752 -0.03270521 -0.0227165  -0.00747272]",x-position of midpoint of spar cap on lower surface for strain calculation
# rotorse.yu_strain_spar,"[-2.68994463 -2.6714235  -2.59877645 -2.59876147 -2.63808801 -2.55904653
#  -2.44879187 -2.46592522 -2.41897788 -2.3721729  -2.33443482 -2.26927189
#  -2.19455422 -2.12028532 -2.04105988 -1.95954551 -1.86181616 -1.76368786
#  -1.66177813 -1.56029523 -1.46038569 -1.36359875 -1.26859831 -1.17583691
#  -1.09181366 -1.00796706 -0.92424035 -0.86519809 -0.72733185 -0.2912122 ]",y-position of midpoint of spar cap on upper surface for strain calculation
# rotorse.yl_strain_spar,"[-2.68994463 -2.6714235  -2.59877645 -2.59876147 -2.63808801 -2.55904653
#  -2.44879187 -2.46592522 -2.41897788 -2.3721729  -2.33443482 -2.26927189
#  -2.19455422 -2.12028532 -2.04105988 -1.95954551 -1.86181616 -1.76368786
#  -1.66177813 -1.56029523 -1.46038569 -1.36359875 -1.26859831 -1.17583691
#  -1.09181366 -1.00796706 -0.92424035 -0.86519809 -0.72733185 -0.2912122 ]",y-position of midpoint of spar cap on lower surface for strain calculation

#==================== ======== =====================================
# Read the outputs

max_strain = schema["constraints"]["blade"]["strains_spar_cap_ss"]["max"]

WISDEMout = folder_arch + "/blade_out.npz"

fig3, ax3 = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
fig4, ax4 = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

with np.load(WISDEMout) as a:

    r = np.array(a["rotorse.rs.z_az_m"])
    r = (r-r[0])/(r[-1]-r[0])
    
    #My passed loading stuff:
    data3 = a["rotorse.rs.extreme_strains.F3_N"]  #should be = to my input
    data4 = a["rotorse.rs.extreme_strains.M1_N*m"]  #should be = to my input (well, my input but set in principal axes)
    data5 = a["rotorse.rs.extreme_strains.M2_N*m"]  #should be = to my input (well, my input but set in principal axes)
    data1 = a["rotorse.rs.extreme_strains.strainU_spar"] / max_strain #rebuild the failure constraint
    data2 = a["rotorse.rs.extreme_strains.strainL_spar"] / max_strain #rebuild the failure constraint
    
    # #original gust stuff:
    # data = a["rotorse.rs.strains.F3_N"]
    # data = a["rotorse.rs.strains.M1_N*m"]
    # data = a["rotorse.rs.strains.M2_N*m"]
    # data1 = a["rotorse.rs.strains.strainU_spar"] / max_strain
    # data2 = a["rotorse.rs.strains.strainL_spar"] / max_strain

    # r = range(4)
    # data = a["rotorse.rs.constr.constr_max_strainU_spar"]
    # data = a["rotorse.rs.constr.constr_max_strainL_spar"]

    ax3.plot(r,-data1,'o-', label=f'SS')  
    ax3.plot(r, data2,'x-', label=f'PS') #NEGATIVE to make them both positive??? but the strain should be > 0 on the PS!!??

    ax4.plot(r,data3,'-', label=f'F3')
    ax4.plot(r,data4,'-', label=f'M1')
    ax4.plot(r,data5,'-', label=f'M2')

ax3.legend()
ax3.set_xlabel("r/R")
ax4.legend()
ax4.set_xlabel("r/R")

plt.show()