from multiprocessing.sharedctypes import Value
import os
import yaml, ast

from wisdem.inputs import load_yaml, write_yaml #, validate_without_defaults, validate_with_defaults, simple_types
from wisdem.inputs import write_geometry_yaml

# import sys, shutil
import numpy as np
import matplotlib.pyplot as plt

# need to use the HiFi layup so we can convert the thickenss dv into the infividual thicknesses of the layers
# #NOTE: if this import fails, please consider doing: `export PYTHONPATH=$PYTHONPATH:/path/to/ATLANTIS_UM-BYU_utils/SETUP` 
import material_info

def my_write_yaml(instance, foutput):
    if os.path.isfile(foutput):
        print(f"File {foutput} already exists... replacing it.")
        os.remove(foutput)
    # Write yaml with updated values
    with open(foutput, "w", encoding="utf-8") as f:
        yaml.dump(instance, f)

#==================== DEFINITIONS  =====================================

    """ This is a semi automatic script to map the DVs of the HiFi model onto the LoFi model. 
    
    The inputs of this script are:
    - `wt_input`: an input Low Fidelity model under the form of a yaml turbine (WindIO format)
    - `DV_input`: a DVCentresCon.dat (output of 'modified' TACS) that describes the location and thickness
        of all structural panels in the model, and also gives the value of the constrsint in each 
        panel (corresponding to the loading that was used as an input to TACS).

    The output of this script is:
    - `wt_output`: an output Low Fidelity model (yaml). It copies most of the input turbine, but replaces the structural
      information (spanwaise position of the laminates and their thickness) with the information mapped
      from the high fidelity model. This file will be written in the current dir.

    Optionally, this script can also:
    - plot the distribution of the panel thickness and of whatever constraint value was given in the DVCentresCon file (see `doPlots`)
    - write an alternate DVGroup file that can be used together with TACS to produce colored visual of the blade model. 
        A different color is given to every group of laminate along the chordwise direction. This is mostly to verify that 
        The mapping between lofi laminates and hifi panels is consistent (see `writeDVGroupForColor`).
    - write a `mappingFile` that gives the static correpondance between each hifi panel (their index) and the region on the lofi
        model. It does the same thing to map each DV to lofi regions (see `writeDVGroupMapping`).

    ## Modeling note:
    Currently, there needs to be as many hifi panels around the airfoil than there are zones in the lofi model.
    They are mapped with respect to their order from TE SS to TE PS.
    """


## ======================= File management and inputs =================================================
mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file

wt_input = "Madsen2019_composite_v02.yaml" # the reference turbine yaml
bladeInput = "composite"

# --PART I-- 
# Tranfering HiFi panel distribution to LoFi model

# Example set of options:
ylab = "failure" #a descriptor of under what load condition the DV_input file was obtained.
DV_input = "FatigueWithTacs_DVCentresCon.dat" #the output of a hifi structural analysis.
DV_folder = "/Users/dcaprace/Documents/BYU/devel/Python/WEIS/examples_BYU/09_compare_struct/2pt_beam_structonly_6173636_1pt_beam/Solutions/FatigueWithTacs__0" #location where to find the DV)input file(s)
wt_output = "tmp_composite_model" #name of the output turbine of this script (i.e., the wt_input modified with DV_input)

# #Original constant thickness model, under nominal loads
# ylab = "nominal" 
# DV_input = "/Users/dg/Documents/BYU/simulation_data/ATLANTIS/MDAO/Structural/Struct_solutions_nominal_iso_L2_3/aeroload_DVCentresCon.dat"
# DV_folder = ""
# wt_output = "Madsen2019_10_forWEIS_isotropic_ED"

# #Original constant thickness model, under nominal loads, WITH gravity in +y
# ylab = "nominalg" 
# DV_input = "/Users/dg/Documents/BYU/simulation_data/ATLANTIS/MDAO/Structural/Struct_solutions_nominal_iso_L2_4/aeroload_DVCentresCon.dat"
# DV_folder = ""
# wt_output = "Madsen2019_10_forWEIS_isotropic_ED"

# #Original constant thickness model, under DEL
# ylab = "damage" 
# DV_input = "/Users/dg/Documents/BYU/simulation_data/ATLANTIS/MDAO/Structural/Struct_solutions_DEL1.1Scaled_L2_4/aeroload_DVCentresCon.dat"
# DV_folder = ""
# wt_output = "Madsen2019_10_forWEIS_isotropic_ED"

# --PART II--
# Optional plots and processing
# Note: the DV Groups files describe how panels are grouped together to form the DVs.

# Example set of options:
DVGroup_input = "FatigueWithTacs_DVGroupCentres.dat" #only used if writeDVGroupForColor or writeDVGroupMapping=True
DVGroup_output = "HiFi_DVs" + os.sep + "generic_DVGroupCentres_colors.dat" #only used if writeDVGroupForColor=True



## ======================= Modeling details =================================================

if bladeInput == "aluminum":
    #from TE_SS to TE_PS: list of the structural zones
    skinLoFi = ["DP17_DP15","DP15_DP13","DP13_DP10_uniax","DP10_DP09","DP09_DP08","DP08_DP07","DP07_DP04_uniax","DP04_DP02","DP02_DP00",]
    websLoFi = ["Web_fore_panel","Web_aft_panel","Web_te_panel"] #the name they have in the LAYER section **NOT** the WEB section!
    leHiFi = "SPAR.04" #identifier of the leading edge in the hifi descriptor
    ssHiFi = "U_" #identifier of the suction side in the hifi descriptor
    psHiFi = "L_" #identifier of the pressure side in the hifi descriptor
    websHiFi = ["SPAR.03","SPAR.02","SPAR.01"]
    
    nDVcopies = 1 #one DV per group of panels
elif bladeInput == "composite":
    #from TE_SS to TE_PS: list of the structural zones
    skinLoFi = [["DP17_DP15_triax","DP17_DP15_uniax","DP17_DP15_balsa"],
                ["DP15_DP13_triax","DP15_DP13_uniax","DP15_DP13_balsa"],
                ["DP13_DP10_triax","DP13_DP10_uniax","DP13_DP10_balsa"],
                ["DP10_DP09_triax","DP10_DP09_uniax","DP10_DP09_balsa"],
                ["DP09_DP08_triax","DP09_DP08_uniax","DP09_DP08_balsa"],
                ["DP08_DP07_triax","DP08_DP07_uniax","DP08_DP07_balsa"],
                ["DP07_DP04_triax","DP07_DP04_uniax","DP07_DP04_balsa"],
                ["DP04_DP02_triax","DP04_DP02_uniax","DP04_DP02_balsa"],
                ["DP02_DP00_triax","DP02_DP00_uniax","DP02_DP00_balsa"]]
    websLoFi = [["Web_fore_biax","Web_fore_balsa",],
                ["Web_aft_biax","Web_aft_balsa",],
                ["Web_te_triax","Web_te_balsa",]] #the name they have in the LAYER section **NOT** the WEB section!
    leHiFi = "SPAR.04" #identifier of the leading edge in the hifi descriptor
    ssHiFi = "U_" #identifier of the suction side in the hifi descriptor
    psHiFi = "L_" #identifier of the pressure side in the hifi descriptor
    websHiFi = ["SPAR.03","SPAR.02","SPAR.01"]

    nDVcopies = 3 #one DV per three groups of panels, because we duplicate the DV to each blade

    #WHAT TO DO WITH THE TE SPAR? We can probably neglect it because it is almost the same as connecting the skin with the same properties in DP00
else:
    ValueError("I don't know this blade.")


ncPanelHiFi = 9 #number of panels over the airfoil in hifi model

# TE_width_percentage = .05 #assuming a constant percentage along the chord: this is just to avoid having to compute the chord length corresponding to the width measured along the airfoil

span_dir = 1 #0=x,1=y,2=z
chord_dir = 2 #0=x,1=y,2=z

R = 89.166 #TODO: get that from wt_input
R0 = 2.8 #TODO: get that from wt_input

trans_len = 0.002 #length of the transition between panels in the lofi model, in %(R-R0)


## ======================= More script options =================================================

doPlots = False
debug = True
writeDVGroupForColor = False 
writeDVGroupMapping = True

# for plots:
spars = ["DP07_DP04_uniax",
         "DP13_DP10_uniax"]  #names of the spars in the lofi model
spars_legend = ["PS","SS"]  #corresponding name that will be displayed on the plots

# for DVGoupMapping
mappingFile = mydir + os.sep + wt_output + os.sep + 'hifiMapping.yaml'


## ======================= =============== =================================================
## ======================= =============== =================================================
## ======================= =============== =================================================
# NO FURTHER USER MODIFICATION SHOULD BE REQUIRED FROM THE USER AFTER THIS POINT

#==================== LOAD HiFi DVs DATA =====================================

DV_file = DV_folder + os.sep + DV_input
ncon = 0

#Read the constitutive component file
with open(DV_file, 'r') as f:
    lines = f.readlines()

    nentry = len(lines[0].split(" "))
    ncon = nentry-6

    HiFiDVs_idx = np.zeros(len(lines))
    HiFiDVs_pos = np.zeros((len(lines),3))
    HiFiDVs_thi = np.zeros(len(lines))
    HiFiDVs_con = np.zeros((len(lines), ncon))
    HiFiDVs_des = [None]*len(lines)

    i = 0
    for line in lines:
        buff = line.split(" ")
        HiFiDVs_idx[i]  = int(buff[0])
        HiFiDVs_pos[i,:] = [ float(b) for b in buff[1:4] ] #pos
        HiFiDVs_thi[i] = 0.09 #float(buff[4]) #dv
        HiFiDVs_con[i,:] = [ float(b) for b in buff[5:-1] ] #constraints
        HiFiDVs_des[i] = buff[-1]
        i+=1


#Identify the number of non-identical y position in the hifi data
# assuming they are organized in sections

nid = np.size(HiFiDVs_pos,0)
thr = 1e-3 #distance threshorld.
yhf_web = []
yhf_skn = []

for i in range(nid):
    if any([cmp in HiFiDVs_des[i] for cmp in websHiFi]):
        if all( abs(yhf_web - HiFiDVs_pos[i,span_dir]) > thr ) and HiFiDVs_pos[i,span_dir] >= R0:
            yhf_web.append(HiFiDVs_pos[i,span_dir])
    elif any([cmp in HiFiDVs_des[i] for cmp in [leHiFi,ssHiFi,psHiFi]]):
        if all( abs(yhf_skn - HiFiDVs_pos[i,span_dir]) > thr ) and HiFiDVs_pos[i,span_dir] >= R0:
            yhf_skn.append(HiFiDVs_pos[i,span_dir])
    else:
        print(f"WARNING: no group found for id {i} with descr {HiFiDVs_des[i]}")

yhf_web = np.sort(yhf_web)
yhf_web_oR = (yhf_web - R0)/(R-R0)
nhf_web = len(yhf_web)
yhf_skn = np.sort(yhf_skn)
yhf_skn_oR = (yhf_skn - R0)/(R-R0)
nhf_skn = len(yhf_skn)

if debug:
    print("Spanwise hf web locations:")
    print(yhf_web)
    print("Spanwise hf skin locations:")
    print(yhf_skn)

#Allocate a mapping object: for each constitutive element, gather the spanwise index, and the name of the corresponding lofi zone 
HiFiDVs_mapping = [{} for i in range(len(HiFiDVs_thi))]

#==================== LOAD TURBINE =====================================

fname_wt_input = mydir + os.sep + wt_input
turbine = load_yaml(fname_wt_input) 

Rc = turbine["components"]["blade"]["outer_shape_bem"]["chord"]["grid"]
chord = turbine["components"]["blade"]["outer_shape_bem"]["chord"]["values"]
twist = turbine["components"]["blade"]["outer_shape_bem"]["twist"]["values"] #assuming it uses the same grid
p_ax = turbine["components"]["blade"]["outer_shape_bem"]["pitch_axis"]["values"] #assuming it uses the same grid


#==================== PROCESS INTERNAL REPRESENTATION OF LOFI STRUCTURE =====================================

lay_ls = turbine["components"]["blade"]["internal_structure_2d_fem"]["layers"]

# =========================================================================
# =================== PART I: HiFi panels to LoFI   =======================
# =========================================================================

#==================== INTERNAL REPRESENTATION OF HIFI STRUCTURE  =====================================

# --------- skin ------------

if len(skinLoFi) != ncPanelHiFi:
    raise ValueError("Can't match lofi and hifi representations because they use different number of panels.")

skin_hifi = np.nan*np.empty((nhf_skn,ncPanelHiFi,4))
skin_hifi[:,0,1] = 0 #the thickness
skin_hifi[:,0,2] = 0 #the index of the component mapped to this specific panel
skin_hifi[:,0,3] = 0.

skin_hifi_con = np.nan*np.empty((nhf_skn,ncPanelHiFi,ncon))

for i in range(nid): #looping over the groups
    if HiFiDVs_pos[i,span_dir] >= R0: #only do blade 1
        #determine my index along the span
        iz = np.where(yhf_skn >= HiFiDVs_pos[i,span_dir] - thr)[0][0]

        #compute the chordwise position of the current panel (precomp convention)
        #XXX: WE NEGLECT TWIST
        c_curr = np.interp(yhf_skn_oR[iz],Rc,chord)
        p_ax_curr = np.interp(yhf_skn_oR[iz],Rc,p_ax)

        loc_c = HiFiDVs_pos[i,chord_dir]/c_curr + p_ax_curr #rough location along the chordwise direction, from 0 to 1
        if psHiFi in HiFiDVs_des[i]:
            loc_s = 1.0 - 0.5*loc_c #rough location along the PS, from 0.5 to 1
        elif ssHiFi in HiFiDVs_des[i]:
            loc_s = 0.5*loc_c #rough location along the SS, from 0.0 to 0.5
        elif leHiFi in HiFiDVs_des[i]:
            loc_s = 0.5 #leading edge
        else:
            #unrecognized component, skip it
            # print(f"skipped: {HiFiDVs_des[i]}")
            continue

        #add the current panel to the list
        ic = np.where(np.isnan(skin_hifi[iz,:,0]))[0]
        if len(ic) == 0:
            print(f"ERROR: could not assign panel {i} at location {yhf_skn_oR[iz]},{loc_s}: layer full already.")
            print(f"   full coords: {HiFiDVs_pos[i,:]}")
            print(f"   local chord: {-c_curr*p_ax_curr}, {c_curr*(1.-p_ax_curr)}")
            print(f"   {skin_hifi[iz,:,0]}")
            continue
        ic = ic[0]
        
        skin_hifi[iz,ic,0] = loc_s
        skin_hifi[iz,ic,1] = HiFiDVs_thi[i]
        skin_hifi[iz,ic,2] = HiFiDVs_idx[i]
        skin_hifi[iz,ic,3] = HiFiDVs_pos[i,span_dir]

        if ncon>0:
            skin_hifi_con[iz,ic,:] = HiFiDVs_con[i,:]

        HiFiDVs_mapping[i]["zone"] = skinLoFi[ic]
        HiFiDVs_mapping[i]["iz"] = iz

#now sort them to follow precomp convension
for j in range(nhf_skn):
    sort_idx = np.argsort(skin_hifi[j,:,0])
    skin_hifi[j,:,:] = skin_hifi[j,sort_idx,:]

if debug:
    print("High fidelity center of panels:")
    print(skin_hifi[:,:,0])
    # print(skin_hifi[:,:,1])

# print("High fidelity mapping to skin panels:")
# print(skin_hifi[:,:,2])

# ------ webs ------------

if len(websLoFi) != len(websHiFi):
    raise ValueError("must have the same number of elements in websLoFi and websHiFi so that they can be matched.")

nwebs = len(websHiFi)

webs_hifi = -np.ones((nhf_web,nwebs,4))
webs_hifi_con = np.nan*np.empty((nhf_web,nwebs,ncon))

for i in range(nid):
    if HiFiDVs_pos[i,span_dir] >= R0: #only do blade 1
        #determine my index along the span
        iz = np.where(yhf_web >= HiFiDVs_pos[i,span_dir] - thr)[0][0]

        #compute the index of the web
        iw = -1
        for j in range(nwebs):
            if websHiFi[j] in HiFiDVs_des[i]:
                iw = j
        if iw==-1:
            #unrecognized component, skip it
            continue

        #leave [0] as is
        webs_hifi[iz,iw,1] = HiFiDVs_thi[i]
        webs_hifi[iz,iw,2] = HiFiDVs_idx[i]
        webs_hifi[iz,iw,3] = HiFiDVs_pos[i,span_dir]

        if ncon>0:
            webs_hifi_con[iz,iw,:] = HiFiDVs_con[i,:]

        HiFiDVs_mapping[i]["zone"] = websLoFi[iw]
        HiFiDVs_mapping[i]["iz"] = iz

if debug:
    print("High fidelity webs:")
    print(webs_hifi[:,:,0]) #note that if there is still -1 in there, this means that the web is not defined at that location



#==================== MATCH INTERNAL REPRESENTATION OF LOFI STRUCTURE AND HIFI DVs =====================================

def compute_edges(yhf_oR):
    # Assuming that the first panel starts at 0 and the last one ends at 1.
    # This might not be the case for webs! That's why we 
    mids = np.zeros(len(yhf_oR)+1)
    mids[-1] = 1.0 

    try:
        #progress from the tip, knowing that the far edge of the last panel is at 1.0
        for i in range(len(yhf_oR)-1,0,-1 ):
            mids[i] = yhf_oR[i] + (yhf_oR[i] - mids[i+1])
            if (yhf_oR[i] - mids[i])<= 0.0: #some tolerance
                raise(ValueError())
    except:
        mids[1:-1] = 0.5 * (yhf_oR[1:] + yhf_oR[:-1])

    mids[0] = 0.0 #forcing 1st point to be 0.
    return mids


def fill_lofi_layers(ylf_oR,lofi_regions,hifi_regions,lofi_layup):
    values = np.zeros(len(ylf_oR))

    cnt = 0
    for ireg,regions in enumerate(lofi_regions):
        for region in np.atleast_1d(regions):
            #based on the name, detect the index 
            layer = []
            for lay in lofi_layup:
                if region in lay["name"]:
                    layer = lay
                    break
            if not layer:
                #skip this: might be a web that I don't want here
                continue

            if debug:
                print(f"Filling skin zone n# {ireg}: {region}")

            if bladeInput == 'composite':
                
                #look for the first non-0 index along the span. Some might be negative if, eg, a given spar only spans part of the span
                idspan = np.where(hifi_regions[:,ireg,2]>=0,)[0]
                if len(idspan)==0:
                    raise ValueError(f"something unexpected happened: region {ireg} has no assignment. Isnt there a extra spar in the LoFi?")
                idspan = idspan[0]

                #find which part of the hifi we are in for materials
                i_glo = np.where(HiFiDVs_idx==hifi_regions[idspan,ireg,2])[0]
                if len(i_glo)==0:
                    raise ValueError(f"Could not find {hifi_regions[0,ireg,2]}")
                i_glo = i_glo[0] #find the index in the global list of descriptors for the current layer (we use the dedcriptor at the root, 0, assuming its all the same along the span)
                
                #inferring the userDescript (see how it's done in setup_tacs)                
                compDescript = HiFiDVs_des[i_glo]
                if 'SPAR' in compDescript:
                    userDescript = "SPARS" 
                else:
                    userDescript = "CHORD_%d"%int(compDescript.split(".")[-1])
   
                _, part, _ = material_info.getPanelCoord(userDescript, [compDescript,])
                if debug:
                    print(f">>> {part}")

                #get the relative layer thickness
                #interpolate the thickness ratio at the correct radius for this region.
                #   (caution: multiple layers may be made of the same material)
                layup, _, _ = material_info.materials_spline(part)

                r_glo = hifi_regions[:,ireg,3]
                weight = np.zeros(nhf_skn)

                my_material = region.split("_")[-1].upper()
                mykey = "frac_"+my_material
                for hifi_layer in layup:
                    if mykey in hifi_layer:
                        weight += np.interp(r_glo,layup["r_start"],layup[hifi_layer])
            else: 
                weight = np.ones(nhf_skn)        

            for j in range(nhf_skn):
                values[2*j] = hifi_regions[j,ireg,1] * weight[j]
                values[2*j+1] = hifi_regions[j,ireg,1] * weight[j]

            layer["thickness"]["grid"] = ylf_oR.tolist()
            layer["thickness"]["values"] = values.tolist()
            cnt += 1

    if type(lofi_regions[0]) is list:
        expected_regions = len([x for sl in lofi_regions for x in sl])
    else:
        expected_regions = len(lofi_regions)
    if cnt != expected_regions:
        print(f"WARNING: I did fill {cnt} skin/web layers, but the expected number is {expected_regions}.")


#because the Hifi has panels, let's try to mimic that in lofi and do a piecewise defined thickness
yhf_web_mids = compute_edges(yhf_web_oR)   
ylf_web_oR = np.zeros(nhf_web*2)
ylf_web_oR[0] = 0.0
ylf_web_oR[-1] = 1.0
for i in range(len(yhf_web_mids)-2):
    ylf_web_oR[2*i+1] = yhf_web_mids[i+1]-trans_len*.5
    ylf_web_oR[2*i+2] = yhf_web_mids[i+1]+trans_len*.5
if any(np.diff(ylf_web_oR) <= 0):
    raise ValueError("ylf_web_oR not monotonically increasing. Try reduce `trans_len`.")

yhf_skn_mids = compute_edges(yhf_skn_oR)   
ylf_skn_oR = np.zeros(nhf_skn*2)
ylf_skn_oR[0] = 0.0
ylf_skn_oR[-1] = 1.0
for i in range(len(yhf_skn_mids)-2):
    ylf_skn_oR[2*i+1] = yhf_skn_mids[i+1]-trans_len*.5
    ylf_skn_oR[2*i+2] = yhf_skn_mids[i+1]+trans_len*.5
if any(np.diff(ylf_skn_oR) <= 0):
    raise ValueError("ylf_skn_oR not monotonically increasing. Try reduce `trans_len`.")


# --------- skin ------------
fill_lofi_layers(ylf_skn_oR,skinLoFi,skin_hifi,lay_ls)

# --------- webs ------------
fill_lofi_layers(ylf_web_oR,websLoFi,webs_hifi,lay_ls)

if debug:
    print("Planform distro:")
    print(skin_hifi[:,:,2])
    print(webs_hifi[:,:,2])


# ==================== SAVE NEW TURBINE =====================================

fname_wt_output = mydir + os.sep + wt_output +'.yaml'
write_geometry_yaml(turbine, fname_wt_output)

os.system('mkdir %s' % (mydir.replace(" ","\ ") + os.sep + wt_output))
outfile = mydir + os.sep + wt_output + os.sep + 'hifiCstr_' + ylab + '.npz'
np.savez(outfile, skinLoFi = skinLoFi, ylf_skn_oR = ylf_skn_oR, skin_hifi_con = skin_hifi_con, nhf_skn = nhf_skn, ncon = ncon, spars = spars, spars_legend = spars_legend)


#==================== Compare DV Plots #====================

if doPlots:
    #------- skin ----------
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 5))

    for isk in range(len(skinLoFi)):
        
        values = np.zeros(len(ylf_skn_oR))
        for j in range(nhf_skn):
            values[2*j] = skin_hifi[j,isk,1]
            values[2*j+1] = skin_hifi[j,isk,1]

        lab = "_".join(np.atleast_1d(skinLoFi[isk])[0].split("_")[0:1])

        hp = ax.plot(ylf_skn_oR,values, '-', label=lab)
            
    ax.set_ylabel("thickness [mm]")
    ax.set_xlabel("r/R")
    plt.legend()

    fig.savefig(mydir + os.sep + wt_output + "/thickness_skin.png")

    #------- webs ----------
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 5))

    for isk in range(len(websLoFi)):
        
        values = np.zeros(len(ylf_web_oR))
        for j in range(nhf_web):
            #the thickness is 0 where the web is not defined.
            values[2*j] = max(webs_hifi[j,isk,1],0.0)
            values[2*j+1] = max(webs_hifi[j,isk,1],0.0)
            
        hp = ax.plot(ylf_web_oR,values, '-', label=websLoFi[isk])
            
    ax.set_ylabel("thickness [mm]")
    ax.set_xlabel("r/R")
    plt.legend()

    fig.savefig(mydir + os.sep + wt_output + "/thickness_webs.png")

    #==================== Compare CONSTRAINTS Plots #====================

    if ncon>0:
        #------- skin ----------
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 5))
        if len(spars)>0:
            fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(15, 5))

        for isk in range(len(skinLoFi)):
            
            values = np.zeros((len(ylf_skn_oR),ncon))
            for c in range(ncon):
                for j in range(nhf_skn):
                    values[2*j,c] = skin_hifi_con[j,isk,c]
                    values[2*j+1,c] = skin_hifi_con[j,isk,c]

            lab = "_".join(np.atleast_1d(skinLoFi[isk])[0].split("_")[0:1])

            hp = ax.plot(ylf_skn_oR,values[:,0], '-', label=lab)
            if ncon>1:
                ax.plot(ylf_skn_oR,values[:,1], '--', color=hp[0].get_color())

            if len(spars)>0:
                # if any( [ sp in skinLoFi[isk] for sp in spars ]):
                #     isp = spars.index(skinLoFi[isk])
                #     hp = ax2.plot(ylf_skn_oR,values[:,0], '-', label=spars_legend[isp], color=hp[0].get_color())
                #     if ncon>1:
                #         ax2.plot(ylf_skn_oR,values[:,0], '--', label=spars_legend[isp], color=hp[0].get_color())
                for isp,sp in enumerate(spars):
                    for skins in np.atleast_1d(skinLoFi[isk]):
                        if sp in skins:
                            hp = ax2.plot(ylf_skn_oR,values[:,0], '-', label=spars_legend[isp], color=hp[0].get_color())
                            if ncon>1:
                                ax2.plot(ylf_skn_oR,values[:,0], '--', label=spars_legend[isp], color=hp[0].get_color())
                                    
        ax.set_ylabel(ylab)
        ax.set_xlabel("r/R")
        ax.legend()
        ax2.set_ylabel(ylab)
        ax2.set_xlabel("r/R")
        ax2.legend()

        fig.savefig(mydir + os.sep + wt_output + f"/{ylab}_skin.png")
        fig2.savefig(mydir + os.sep + wt_output + f"/{ylab}_spars.png")

        #------- webs ----------
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 5))

        for isk in range(len(websLoFi)):

            values = np.zeros((len(ylf_web_oR),ncon))
            for c in range(ncon):
                for j in range(nhf_web):
                    values[2*j,c] = webs_hifi_con[j,isk,c]
                    values[2*j+1,c] = webs_hifi_con[j,isk,c]

            hp = ax.plot(ylf_web_oR,values[:,0], '-', label=websLoFi[isk])
            if ncon>1:
                ax.plot(ylf_web_oR,values[:,1], '--', color=hp[0].get_color())
            
        ax.set_ylabel(ylab)
        ax.set_xlabel("r/R")
        ax.legend()

        fig.savefig(mydir + os.sep + wt_output + f"/{ylab}_webs.png")

    plt.show()


# ==========================================================================
# =================== PART II: DV Groups to Panels   =======================
# ==========================================================================

# Let us now do the mapping between the DVs and the lo-fi.
# Above, we mapped every single pannel with the lofi. The difference
# here is that a single DV can span several panels. This is the main
# reason why we have 2 types of files:
# - DVCentres: detailes about every single panel
# - DVGroupCentres: gives how panels are grouped per DV


# - Read the DV Group input -

if (writeDVGroupForColor or writeDVGroupMapping):

    #Read the group component file
    with open(os.path.join(DV_folder,DVGroup_input), 'r') as f:
        lines = f.readlines()


        HiFiGrp_idx  = np.zeros(len(lines), int)
        HiFiGrp_pos = np.zeros((len(lines),3))
        HiFiGrp_thi = np.zeros(len(lines))
        # HiFiGrp_con = np.zeros((len(lines), ncon))
        HiFiGrp_des = [None]*len(lines)
        HiFiGrp_icmp = [None]*len(lines)

        i = 0
        for line in lines:
            buff = line.split(" ")
            HiFiGrp_idx[i] = buff[0]
            HiFiGrp_pos[i,:] = [ float(b) for b in buff[1:4] ]
            # HiFiGrp_thi[i] = float(buff[4]) 
            # HiFiGrp_con[i,:] = [ float(b) for b in buff[5:-1] ] 
            buff = line.split('"')
            HiFiGrp_des[i] = buff[1]
            HiFiGrp_icmp[i] = buff[3]
            i+=1

    #Allocate a mapping object: 
    #  binds each group number to the list of constitutive elements it covers
    HiFiDVs_mapping = [[] for i in range(len(HiFiGrp_idx)//nDVcopies)]

    jDDL = 0
    for i in range(len(HiFiGrp_idx)):
        if "SPARS1" in HiFiGrp_des[i] or "L_SKIN1" in HiFiGrp_des[i] or "U_SKIN1" in HiFiGrp_des[i]:
            tmp = ",".join(HiFiGrp_icmp[i][1:-1].split())
            icmp = ast.literal_eval('['+tmp+']')
            HiFiDVs_mapping[jDDL] = icmp
            # print(HiFiGrp_des[i])
            # print(i, jDDL, i//nDVcopies)
            jDDL+=1

if debug:
    print(HiFiDVs_mapping)

# - Write the Group COlor output -

if writeDVGroupForColor:

    #Write the constitutive component file
    with open(DVGroup_output, 'w') as f:
        for i in range(len(HiFiGrp_idx)):
            pattern = "%i %6.5e %6.5e %6.5e %6.5e "
            # pattern += "%6.5e "*ncon
            pattern += "%s\n"

            comp_list = HiFiDVs_mapping[i]
            
            #try to find a match for at least 1 component:
            for comp in comp_list:
                #look in which zone of the skin the component is:
                msk = np.where(skin_hifi[:,:,2] == comp)
                id = msk[1] #index along the 2nd dim (chordwise)
                if len(id) ==0: #this might be a spar
                    msk = np.where(webs_hifi[:,:,2] == comp)
                    id = msk[1] + len(skinLoFi) #web index
                    if len(id) !=0: 
                        break
                else:
                    break
            if len(id) ==0: #we did not find anything
                id = -1

            f.write(pattern%(
                HiFiGrp_idx[i],
                HiFiGrp_pos[i,0], HiFiGrp_pos[i,1], HiFiGrp_pos[i,2],
                id,
                # tuple(HiFiDVs_con[i,:]),
                HiFiGrp_icmp[i]
            ))



# - Write the mapping for use in MACH -

if writeDVGroupMapping:

    # I know how panels map to planform, and I know how DVs map to panels.
    # Let's just determine how DVs map to planform.

    # skin  
    ni = len(skin_hifi)
    nj = len(skin_hifi[0])
    skin_DV = [ [-1]*nj for i in range(ni)]

    for i in range(ni):
        for j in range(nj):
            # find what DV (i.e. panel group) the current panel belongs to            
            for ig,group in enumerate(HiFiDVs_mapping):
                if skin_hifi[i,j,2] in group:
                    skin_DV[i][j] = ig

    # webs
    ni = webs_hifi.shape[0]
    nj = webs_hifi.shape[1]
    webs_DV = [ [-1]*nj for i in range(ni)]

    for i in range(ni):
        for j in range(nj):
            # find what DV (i.e. panel group) the current panel belongs to            
            for ig,group in enumerate(HiFiDVs_mapping):
                if webs_hifi[i,j,2] in group:
                    webs_DV[i][j] = ig

    print("WARNING: THE FOLLOWING is NOT correct for the composite model. ")
    print("COPY THIS TO dv2planform")
    print(skin_DV)
    print("COPY THIS TO dv2webs:")
    print(webs_DV)
    print("COPY THIS TO planform_rlocs:")
    print(yhf_skn_mids)
    print("COPY THIS TO webs_rlocs:")
    print(yhf_web_mids)


    # create the dict for output

    mappingDict = {}

    mappingDict["panels2planform"] = np.intc(skin_hifi[:,:,2]).tolist()
    mappingDict["panels2webs"] = np.intc(webs_hifi[:,:,2]).tolist()

    mappingDict["dv2planform"] = skin_DV
    mappingDict["dv2webs"] = webs_DV

    mappingDict["planform_rlocs"] = yhf_skn_mids.tolist()
    mappingDict["webs_rlocs"] = yhf_web_mids.tolist()
    
    write_yaml(mappingDict,mappingFile)
