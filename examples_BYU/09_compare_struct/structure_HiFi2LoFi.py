import os
import yaml

from wisdem.inputs import load_yaml, write_yaml #, validate_without_defaults, validate_with_defaults, simple_types
from wisdem.inputs import write_geometry_yaml

# import sys, shutil
import numpy as np
import matplotlib.pyplot as plt


def my_write_yaml(instance, foutput):
    if os.path.isfile(foutput):
        print(f"File {foutput} already exists... replacing it.")
        os.remove(foutput)
    # Write yaml with updated values
    with open(foutput, "w", encoding="utf-8") as f:
        yaml.dump(instance, f)

#==================== DEFINITIONS  =====================================

    """ This is a semi automatic script to map the DVs of the HiFi model onto the LoFi model.

    Currently, there needs to be as many hifi panels around the airfoil than there are zones in the lofi model.
    They are mapped with respect to their order from TE SS to TE PS.
    """

## File management
mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file

# wt_input = "Madsen2019_10_forWEIS_isotropic.yaml" 
wt_input = "Madsen2019_10_forWEIS_isotropic.yaml" 
wt_output = "Madsen2019_10_forWEIS_isotropic_TEST.yaml"

DV_input = "aeroload_DVCentres.dat"
DV_input = "aeroload_DVCentresCon.dat" #from a structural analysis under nominal loads
# DV_input = "Fatigue_force_allwalls_L2_DEL_neq1_0DVCentresCon.dat" #(partial) result of an optimization under DEL
# DV_input = "TMP/aeroload_DVCentresCon.dat" #using IC from the case above, and nominal loads
# DV_input = "glo_iter1/Fatigue_force_allwalls_L2_DEL_neq1_0DVCentresCon.dat" #full result of an optimization under DEL
DV_folder = mydir + os.sep + "HiFi_DVs"

#Original constant thickness model, under DEL
DV_input = "/Users/dg/Documents/BYU/simulation_data/ATLANTIS/MDAO/Structural/Struct_solutions_DEL1.1Scaled_L2_4/aeroload_DVCentresCon.dat"
DV_folder = ""

# #Optimized 1st iter model, under DEL
# DV_input = "/Users/dg/Documents/BYU/simulation_data/ATLANTIS/MDAO/Aerostructural/Optimization/1pt_fatigue_44949859_L3/Fatigue_force_allwalls_L2_DEL_neq1_0DVCentresCon.dat"
# DV_folder = ""
# wt_output = "Madsen2019_10_forWEIS_isotropic_DEL_ITER1.yaml"


#from TE_SS to TE_PS: list of the structural zones
skinLoFi = ["DP17_DP15","DP15_DP13","DP13_DP10_uniax","DP10_DP09","DP09_DP08","DP08_DP07","DP07_DP04_uniax","DP04_DP02","DP02_DP00",]
websLoFi = ["Web_fore_panel","Web_aft_panel","Web_te_panel"] #the name they have in the LAYER section **NOT** the WEB section!
leHiFi = "SPAR.04" #identifier of the leading edge in the hifi descriptor
ssHiFi = "U_" #identifier of the suction side in the hifi descriptor
psHiFi = "L_" #identifier of the pressure side in the hifi descriptor
websHiFi = ["SPAR.03","SPAR.02","SPAR.01"]


ncPanelHiFi = 9 #number of panels over the airfoil in hifi model

TE_width_percentage = .05 #assuming a constant percentage along the chord: this is just to avoid having to compute the chord length corresponding to the width measured along the airfoil

span_dir = 1 #0=x,1=y,2=z
chord_dir = 2 #0=x,1=y,2=z

R = 89.166
R0 = 2.8

trans_len = 0.01 #length of the transition between panels in the lofi model, in %(R-R0)

doPlot = True
debug = False

spars = ["DP07_DP04_uniax","DP13_DP10_uniax"] 
spars_legend = ["PS","SS"]

#==================== LOAD HiFi DVs DATA =====================================

DV_file = DV_folder + os.sep + DV_input
ncon = 0

#Read the constitutive component file
with open(DV_file, 'r') as f:
    lines = f.readlines()

    nentry = len(lines[0].split(" "))
    ncon = nentry-6

    HiFiDVs_pos = np.zeros((len(lines),3))
    HiFiDVs_thi = np.zeros(len(lines))
    HiFiDVs_con = np.zeros((len(lines), ncon))
    HiFiDVs_des = [None]*len(lines)

    i = 0
    for line in lines:
        buff = line.split(" ")
        HiFiDVs_pos[i,:] = [ float(b) for b in buff[1:4] ]
        HiFiDVs_thi[i] = float(buff[4]) 
        HiFiDVs_con[i,:] = [ float(b) for b in buff[5:-1] ] 
        HiFiDVs_des[i] = buff[-1]
        i+=1


#Identify the number of non-identical y position in the hifi data
# assuming they are organized in sections

nid = np.size(HiFiDVs_pos,0)
thr = 1e-3 #distance threshorld. If closer than that in 
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

# ------ skin ------------

lay_ls = turbine["components"]["blade"]["internal_structure_2d_fem"]["layers"]

# #create a matrix where we store the bounds of each zone in the lofi model
# skin_bnds = -np.ones((nhf_skn,len(skinLoFi)+1))
# skin_bnds[:,0] = 0.0 #TE, SS
# skin_bnds[:,-1] = 1.0 #TE, PS

# #first pass: take advantage of start_nd_arc
# for layer in lay_ls:
#     #based on the name, detect the index 
#     try:
#         ilay = skinLoFi.index(layer["name"])
#     except ValueError:
#         #skip this: might be a web that I don't want here
#         continue
    
#     if "start_nd_arc" in layer and "fixed" not in layer["start_nd_arc"]:
#         skin_bnds[:,ilay] = np.interp(yhf_skn_oR,layer["start_nd_arc"]["grid"],layer["start_nd_arc"]["values"])

# #second pass: take advantage of end_nd_arc but do not overwrite
# for layer in lay_ls:
#     #based on the name, detect the index 
#     try:
#         ilay = skinLoFi.index(layer["name"])
#     except ValueError:
#         #skip this: might be a web that I don't want here
#         continue
    
#     if "end_nd_arc" in layer and  "fixed" not in layer["end_nd_arc"] and skin_bnds[0,ilay+1] <0:
#         skin_bnds[:,ilay+1] = np.interp(yhf_skn_oR,layer["end_nd_arc"]["grid"],layer["end_nd_arc"]["values"])

# #third pass: those where width is defined have a fixed width wrt given 
# #CAUTION: width is dimensional and measured along the path of the airfoil
# for layer in lay_ls:
#     #based on the name, detect the index 
#     try:
#         ilay = skinLoFi.index(layer["name"])
#     except ValueError:
#         #skip this: might be a web that I don't want here
#         continue
    
#     if "width" in layer and  "start_nd_arc" in layer and skin_bnds[0,ilay+1] <0:
#         skin_bnds[:,ilay+1] = .5*TE_width_percentage #*np.interp(yhf_skn_oR,Rc,chord)

#     if "width" in layer and  "end_nd_arc" in layer and skin_bnds[0,ilay] <0:
#         skin_bnds[:,ilay] = 1.0-.5*TE_width_percentage #*np.interp(yhf_skn_oR,Rc,chord)

# # CHECK: verify that there is no gap in the matrix
# for i in range(len(skinLoFi)-1):
#     if skin_bnds[0,i+1]<0:
#         print(f"ERROR: missing information on the boundary between {i}:{skinLoFi[i]} and {i+1}:{skinLoFi[i+1]}")
#     if any( skin_bnds[:,i] > skin_bnds[:,i+1]):
#         print(f"ERROR: problem in the geometry. Bounds are not linearly increasing between {i} and {i+1}")

# print("Low fidelity layer bounds:")
# print(skin_bnds)



#==================== INTERNAL REPRESENTATION OF HIFI STRUCTURE  =====================================

# --------- skin ------------

if len(skinLoFi) != ncPanelHiFi:
    raise ValueError("Can't match lofi and hifi representations because they use different number of panels.")

skin_hifi = np.nan*np.empty((nhf_skn,ncPanelHiFi,2))
skin_hifi[:,0,1] = 0

skin_hifi_con = np.nan*np.empty((nhf_skn,ncPanelHiFi,ncon))

for i in range(nid):
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



# ------ webs ------------

if len(websLoFi) != len(websHiFi):
    raise ValueError("must have the same number of elements in websLoFi and websHiFi so that they can be matched.")

nwebs = len(websHiFi)

webs_hifi = -np.ones((nhf_web,nwebs))
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

        webs_hifi[iz,iw] = HiFiDVs_thi[i]

        if ncon>0:
            webs_hifi_con[iz,iw,:] = HiFiDVs_con[i,:]

        HiFiDVs_mapping[i]["zone"] = websLoFi[iw]
        HiFiDVs_mapping[i]["iz"] = iz

if debug:
    print("High fidelity webs:")
    print(webs_hifi[:,:]) #note that if there is still -1 in there, this means that the web is not defined at that location



#==================== MATCH INTERNAL REPRESENTATION OF LOFI STRUCTURE AND HIFI DVs =====================================

#because the Hifi has panels, let's try to mimic that in lofi and do a piecewise defined thickness
yhf_web_mids = 0.5 * (yhf_web_oR[1:] + yhf_web_oR[:-1])
ylf_web_oR = np.zeros(nhf_web*2)
ylf_web_oR[0] = 0.0
ylf_web_oR[-1] = 1.0
for i in range(len(yhf_web_mids)):
    ylf_web_oR[2*i+1] = yhf_web_mids[i]-trans_len*.5
    ylf_web_oR[2*i+2] = yhf_web_mids[i]+trans_len*.5
if any(np.diff(ylf_web_oR) <= 0):
    raise ValueError("ylf_web_oR not monotonically increasing. Try reduce `trans_len`.")

yhf_skn_mids = 0.5 * (yhf_skn_oR[1:] + yhf_skn_oR[:-1])
ylf_skn_oR = np.zeros(nhf_skn*2)
ylf_skn_oR[0] = 0.0
ylf_skn_oR[-1] = 1.0
for i in range(len(yhf_skn_mids)):
    ylf_skn_oR[2*i+1] = yhf_skn_mids[i]-trans_len*.5
    ylf_skn_oR[2*i+2] = yhf_skn_mids[i]+trans_len*.5
if any(np.diff(ylf_skn_oR) <= 0):
    raise ValueError("ylf_skn_oR not monotonically increasing. Try reduce `trans_len`.")


# --------- skin ------------
values = np.zeros(len(ylf_skn_oR))

cnt = 0
for layer in lay_ls:
    #based on the name, detect the index 
    try:
        ilay = skinLoFi.index(layer["name"])
        if debug:
            print(f"Filling skin zone n# {ilay}: {layer['name']}")
    except ValueError:
        #skip this: might be a web that I don't want here
        continue
    
    for j in range(nhf_skn):
        values[2*j] = skin_hifi[j,ilay,1]
        values[2*j+1] = skin_hifi[j,ilay,1]

    layer["thickness"]["grid"] = ylf_skn_oR.tolist()
    layer["thickness"]["values"] = values.tolist()
    cnt += 1

if cnt != len(skinLoFi):
    print(f"WARNING: I did only fill {cnt} skin layers, but the expected number is {len(skinLoFi)}.")


# --------- webs ------------
values = np.zeros(len(ylf_web_oR))

cnt = 0
for layer in lay_ls:
    #based on the name, detect the index 
    try:
        ilay = websLoFi.index(layer["name"])
        if debug:
            print(f"Filling web n# {ilay}: {layer['name']}")
    except ValueError:
        #skip this: might be a web that I don't want here
        continue
    
    for j in range(nhf_web):
        #the thickness is 0 where the web is not defined.
        values[2*j] = max(webs_hifi[j,ilay],0.0)
        values[2*j+1] = max(webs_hifi[j,ilay],0.0)

    layer["thickness"]["grid"] = ylf_web_oR.tolist()
    layer["thickness"]["values"] = values.tolist()
    cnt += 1

if cnt != nwebs:
    print(f"WARNING: I did only fill {cnt} webs, but the expected number is {nwebs}.")




# THE FOLLOWING is an attempt to manage the case when the number of panel is different
# NEED to uncomment the section about internal lofi representation

# # table of dictionaries: will count the number of hifi panels in each zone of the lofi model
# th_hf = [[{} for i in range(len(skinLoFi))] for j in range(nhf_skn)]
            
# for i in range(nid):
#         #determine my index along the span
#         iz = np.where(yhf_skn >= HiFiDVs_pos[i,span_dir] - thr)[0][0]

#         #compute the chordwise position of the current panel (precomp convention)
#         #XXX: WE NEGLECT TWIST
#         c_curr = np.interp(yhf_skn_oR[iz],Rc,chord)
#         p_ax_curr = np.interp(yhf_skn_oR[iz],Rc,p_ax)

#         loc_c = HiFiDVs_pos[i,chord_dir]/c_curr + p_ax_curr #rough location along the chordwise direction, from 0 to 1
#         if psHiFi in HiFiDVs_des[i]:
#             loc_s = 1.0 - 0.5*loc_c #rough location along the PS, from 0.5 to 1
        # elif ssHiFi in HiFiDVs_des[i]:
        #     loc_s = 0.5*loc_c #rough location along the SS, from 0.0 to 0.5
        # elif leHiFi in HiFiDVs_des[i]:
        #     loc_s = 0.5*loc_c #leading edge, assimilate to U
        # else:
        #     #unrecognized component, skip it
        #     # print(f"skipped: {HiFiDVs_des[i]}")
        #     continue

#         #add the current panel to the count of panels
#         ic = np.where(loc_s<= skin_bnds[iz,:])[0]
#         if len(ic) == 0:
#             print(f"WARNING: could not assign panel {i} at location {yhf_skn_oR[iz]},{loc_s}.")
#             print(f"   full coords: {HiFiDVs_pos[i,:]}")
#             print(f"   local chord: {-c_curr*p_ax_curr}, {c_curr*(1.-p_ax_curr)}")
#             continue
#         ic = ic[0]
        
#         th_hf[iz][ic][i] = HiFiDVs_thi[i]
    
# print(th_hf)
#==================== UPDATE BLADE DATA =====================================

# turbine["nominal"] = {}
# turbine["nominal"]["description"] = f"nominal loads obtained for inflow velocity {run_settings['U']}"
# turbine["nominal"]["grid_nd"] = locs.tolist()
# turbine["nominal"]["Fn"] = data_a_avg[0,:,ifi].tolist()
# turbine["nominal"]["Ft"] = data_a_avg[1,:,ifi].tolist()




# -  name: DP04_DP02_balsa
#     start_nd_arc:
#         fixed: DP07_DP04_uniax
#         grid: [0.0, 0.034482758620689655, 0.06896551724137931, 0.10344827586206896, 0.13793103448275862, 0.1724137931034483, 0.20689655172413793, 0.24137931034482757, 0.27586206896551724, 0.3103448275862069, 0.3448275862068966, 0.3793103448275862, 0.41379310344827586, 0.4482758620689655, 0.48275862068965514, 0.5172413793103449, 0.5517241379310345, 0.5862068965517241, 0.6206896551724138, 0.6551724137931034, 0.6896551724137931, 0.7241379310344828, 0.7586206896551724, 0.7931034482758621, 0.8275862068965517, 0.8620689655172413, 0.896551724137931, 0.9310344827586207, 0.9655172413793103, 1.0]
#         values: [0.8330699018111011, 0.833946846201656, 0.8413295038792943, 0.829963217753461, 0.8131875066486137, 0.8068177674814875, 0.8021808617108501, 0.7891902310670056, 0.7767293158674035, 0.767925737231656, 0.7606098091220218, 0.7573590217732178, 0.7552333010607861, 0.754022245552566, 0.752745746123007, 0.7515444959802141, 0.7510606017618957, 0.7509998371181033, 0.7514882754615161, 0.7523806499582799, 0.7535425521178344, 0.7548428350274891, 0.7563483800932084, 0.7580534879070009, 0.7595773213458685, 0.7614272985236008, 0.763661388703637, 0.7681054076186038, 0.7777502284081057, 0.7078953787930481]
#     end_nd_arc:
#         fixed: DP02_DP00_uniax
#         grid: [0.0, 0.034482758620689655, 0.06896551724137931, 0.10344827586206896, 0.13793103448275862, 0.1724137931034483, 0.20689655172413793, 0.24137931034482757, 0.27586206896551724, 0.3103448275862069, 0.3448275862068966, 0.3793103448275862, 0.41379310344827586, 0.4482758620689655, 0.48275862068965514, 0.5172413793103449, 0.5517241379310345, 0.5862068965517241, 0.6206896551724138, 0.6551724137931034, 0.6896551724137931, 0.7241379310344828, 0.7586206896551724, 0.7931034482758621, 0.8275862068965517, 0.8620689655172413, 0.896551724137931, 0.9310344827586207, 0.9655172413793103, 1.0]
#         values: [0.9526656851917246, 0.9519933233531207, 0.9410620101717578, 0.9346181290183674, 0.9359289942103652, 0.9404148882998287, 0.9435601806216652, 0.941868780683788, 0.9391549258432071, 0.9370450997258035, 0.9349684314677672, 0.933583164641312, 0.9316870884867144, 0.929282082752966, 0.9263789778631423, 0.9232550601038872, 0.9194261777961372, 0.915390305177415, 0.9109577544669449, 0.9061271070847642, 0.9007367940230925, 0.8947728517210427, 0.8880689171600296, 0.8804520679162664, 0.8724707670376467, 0.865323588113409, 0.8612953528371525, 0.8612900458884405, 0.8994687572731529, 0.960525268939701]

# components:
#     blade:
#         internal_structure_2d_fem:
#             webs:
#                -  name: fore_web
#                   offset_y_pa:
#                       grid: &grid_webs [0.030, 0.125, 0.250, 0.375, 0.500, .625, 0.750, 0.875, 0.925]
#                       values: [-0.55, -0.70, -0.65, -0.50, -0.35, -0.30, -0.25, -0.20, -0.18]
#                -  name: rear_web
#                   offset_y_pa:
#                       grid: *grid_webs
#                       values: [0.55, 0.35, 0.30, 0.30, 0.35, 0.30, 0.25, 0.18, 0.15]
#                -  name: web_te
#                   offset_y_pa:
#                       grid: &grid_te_web [0.210, 0.375, 0.500, .625, 0.750, 0.875, 0.925]
#                       values: [2.50, 2.45, 2.20, 1.80, 1.45, 1.05, 0.92]
#             layers:
#                -  name: DP13_DP10 #upper spar (keep this name so that don't need to change modeling file)
#                   thickness:
#                       grid: &grid_struct2 [0.000, 0.125, 0.250, 0.375, 0.500, .625, 0.750, 0.875, 1.000]
#                       values: [0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009]
#                -  name: DP17_DP15
#                   thickness:
#                       grid: *grid_struct2
#                       values: [0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009]                      
#                -  name: DP15_DP13
#                   thickness:
#                       grid: *grid_struct2
#                       values: [0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009]                      
#                -  name: DP07_DP04 #upper spar (keep this name so that don't need to change modeling file)
#                   thickness:
#                       grid: *grid_struct2
#                       values: [0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009]               
#                -  name: DP09_DP08
#                   material: alu_2024
#                   thickness:
#                       grid: *grid_struct2
#                       values: [0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009]                      
#                -  name: DP10_DP09
#                   thickness:
#                     grid: *grid_struct2
#                     values: [0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009]                      
#                -  name: DP08_DP07
#                   thickness:
#                       grid: *grid_struct2
#                       values: [0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009]                      
#                -  name: DP02_DP00
#                   thickness:
#                       grid: *grid_struct2
#                       values: [0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009]                      
#                -  name: DP04_DP02
#                   thickness:
#                       grid: *grid_struct2
#                       values: [0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009]                      

#                -  name: Web_fore_panel
#                   thickness:
#                       grid: &grid_webs_mat [0.030, 0.925]
#                       values: [0.008, 0.008]
#                -  name: Web_aft_panel
#                   thickness:
#                       grid: *grid_webs_mat
#                       values: [0.010, 0.010]
#                -  name: Web_te_panel
#                   thickness:
#                       grid: *grid_webs_mat
#                       values: [0.010, 0.010]




# ==================== SAVE NEW TURBINE =====================================

fname_wt_output = mydir + os.sep + wt_output
write_geometry_yaml(turbine, fname_wt_output)



#==================== Compare DV Plots #====================

#------- skin ----------
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 5))

for isk in range(len(skinLoFi)):
    
    values = np.zeros(len(ylf_skn_oR))
    for j in range(nhf_web):
        values[2*j] = skin_hifi[j,isk,1]
        values[2*j+1] = skin_hifi[j,isk,1]

    hp = ax.plot(ylf_skn_oR,values, '-', label=skinLoFi[isk])
        
ax.set_ylabel("thickness [mm]")
ax.set_xlabel("r/R")
plt.legend()


#------- webs ----------
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 5))

for isk in range(len(websLoFi)):
    
    values = np.zeros(len(ylf_web_oR))
    for j in range(nhf_web):
        #the thickness is 0 where the web is not defined.
        values[2*j] = max(webs_hifi[j,isk],0.0)
        values[2*j+1] = max(webs_hifi[j,isk],0.0)
        
    hp = ax.plot(ylf_web_oR,values, '-', label=websLoFi[isk])
        
ax.set_ylabel("thickness [mm]")
ax.set_xlabel("r/R")
plt.legend()



#==================== Compare CONSTRAINTS Plots #====================

if ncon>0:
    #------- skin ----------
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 5))
    if len(spars)>0:
        fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(15, 5))

    for isk in range(len(skinLoFi)):
        
        values = np.zeros((len(ylf_skn_oR),ncon))
        for c in range(ncon):
            for j in range(nhf_web):
                values[2*j,c] = skin_hifi_con[j,isk,c]
                values[2*j+1,c] = skin_hifi_con[j,isk,c]

        hp = ax.plot(ylf_skn_oR,values[:,0], '-', label=skinLoFi[isk])
        if ncon>1:
            ax.plot(ylf_skn_oR,values[:,1], '--', color=hp[0].get_color())

        if len(spars)>0:
            if any( [ skinLoFi[isk] in sp for sp in spars ]):
                isp = spars.index(skinLoFi[isk])
                ax2.plot(ylf_skn_oR,values[:,0], '-', label=spars_legend[isp], color=hp[0].get_color())
            
    ax.set_ylabel("failure")
    ax.set_xlabel("r/R")
    ax.legend()
    ax2.set_ylabel("failure")
    ax2.set_xlabel("r/R")
    ax2.legend()


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
        
    ax.set_ylabel("failure")
    ax.set_xlabel("r/R")
    ax.legend()





plt.show()