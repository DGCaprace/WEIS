import os
from wisdem.inputs import load_yaml, write_yaml #, validate_without_defaults, validate_with_defaults, simple_types
from wisdem.inputs import write_geometry_yaml

import numpy as np
import copy

mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file

#original design
wt_input = 'Madsen2019_composite_v02_IC.yaml'
wt_output = 'Madsen2019_composite_v02_IC%s%02d.yaml'

perturbation_list = [   [0,8],
                        [1,7],
                        [2,6],
                        [3,4,5] ]
perturbation_ampl = .02


skinLoFi = [    ["DP17_DP15_triax","DP17_DP15_uniax","DP17_DP15_balsa"],
                ["DP15_DP13_triax","DP15_DP13_uniax","DP15_DP13_balsa"],
                ["DP13_DP10_triax","DP13_DP10_uniax","DP13_DP10_balsa"],
                ["DP10_DP09_triax","DP10_DP09_uniax","DP10_DP09_balsa"],
                ["DP09_DP08_triax","DP09_DP08_uniax","DP09_DP08_balsa"],
                ["DP08_DP07_triax","DP08_DP07_uniax","DP08_DP07_balsa"],
                ["DP07_DP04_triax","DP07_DP04_uniax","DP07_DP04_balsa"],
                ["DP04_DP02_triax","DP04_DP02_uniax","DP04_DP02_balsa"],
                ["DP02_DP00_triax","DP02_DP00_uniax","DP02_DP00_balsa"]]
# websLoFi = [    ["Web_fore_biax","Web_fore_balsa",],
#                 ["Web_aft_biax","Web_aft_balsa",],
#                 ["Web_te_triax","Web_te_balsa",]] #the name they have in the LAYER section **NOT** the WEB section!
  
# ================================================================================
# ================================================================================
# ================================================================================

fname_wt_input = mydir + os.sep + wt_input
turbine = load_yaml(fname_wt_input) 


for i,pids in enumerate(perturbation_list):
    print(f"Perturbation {i}...")
    print(pids)

    turbine_out = copy.deepcopy(turbine)
    lay_ls = turbine_out["components"]["blade"]["internal_structure_2d_fem"]["layers"]

    for pid in pids:

        panel = skinLoFi[pid]
        
        #Looping aver all the layers in the file and save a pointer to those involved in the current panel
        lay_ptrs = []
        for lay in lay_ls:
            if lay["name"] in panel:
                lay_ptrs.append(lay)
        
        #Computing total thickness of the panel
        total_thickness = np.zeros(len(lay_ptrs[0]["thickness"]["values"]))
        for lay in lay_ptrs:
            total_thickness += np.array(lay["thickness"]["values"])

        #modifying thickness while sticking to the fraction
        for lay in lay_ptrs:
            value = np.array(lay["thickness"]["values"]) + perturbation_ampl * np.array(lay["thickness"]["values"])/total_thickness
            lay["thickness"]["values"] = value.tolist()

    fname = wt_output%("p",i+1)
    fname_wt_output = mydir + os.sep + fname
    print(f"...saving to {fname}")
    write_geometry_yaml(turbine_out, fname_wt_output)
