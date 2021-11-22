import os
import sys, shutil
import ast
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import yaml

debug = True #mostly plots and outputs an xdmf file to visualize load vector field (original and scaled)
fakeHiFi = False #use a fake high-fidelity reference loading.

loFiSource = 3
# 1= use a fake lofi distribution
# 2= use a reference static load distribution from low fidelity
# 3= use the aggregated/extrapolated loads from unsteady simulations (see bottom of this file)

method = 2
# 1= scale loads based on a reconstructed  (THIS IS WRONG: nodal values are in N/m2, not N)
# 2= scale the loads using the local value of the known reference and target distributed loads (RECOMMENDED)
# Note: method 1 was an attemps to avoid relying on the hifi "lift" output (the known reference).


fl = ["orig.xmf","scaled.xmf","Fn.png","Ft.png","scaling.png"]

#from ADflow:
def read_force_file(fname):
    
    if fakeHiFi:
        #dummy for test:
        npts = 128
        pos = np.zeros((npts,3))
        forces = np.zeros((npts,3))
        
        for i in range(npts):
            pos[i,1] = (i+1)/npts * 89.166
            forces[i,0] = 5 * i/(npts-1) + 3 * (npts-1-i)/(npts-1)
            forces[i,2] = 1 * i/(npts-1) + 4 * (npts-1-i)/(npts-1)

        if debug:
            f, a = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
            plt.plot(pos[:,1],forces[:,0])
            plt.plot(pos[:,1],forces[:,2])
            
        return pos, forces, [], 0
    
    #---
    pos, forces, conn = [],[],[]
    lw = 16 #width of an entry
    with open(fname,'r') as f:
        l1 = f.readline() 
        buff=l1.split(' ')
        npts = int(buff[0])
        ncell = int(buff[1])

        pos = np.zeros((npts,3))
        forces = np.zeros((npts,3))

        for ip in range(npts):
            l = f.readline()
            k = 0
            pos[ip,0] = float(l[k*lw:(k+1)*lw]) #fixed format
            k+=1
            pos[ip,1] = float(l[k*lw:(k+1)*lw]) #fixed format
            k+=1
            pos[ip,2] = float(l[k*lw:(k+1)*lw]) #fixed format
            k+=1
            forces[ip,0] = float(l[k*lw:(k+1)*lw]) #fixed format
            k+=1
            forces[ip,1] = float(l[k*lw:(k+1)*lw]) #fixed format
            k+=1
            forces[ip,2] = float(l[k*lw:(k+1)*lw]) #fixed format

        conn = f.readlines()

    if debug:
        exportToXdmf("orig.xmf",pos,forces)

    return pos, forces, conn, ncell

#from ADflow:
def getLiftDistribution(testcase):
    dicty = {}
    f = open(testcase, 'r')
    lines = [line.rstrip('\n') for line in f]
    f.close()
    vars_slice = lines[1].replace('Variables = ', '').split('" "')
    #el_line = lines[3].replace('  ZONETYPE=FELINESEG', '') \
    #    .replace('Elements=         ', '').replace(' Nodes =         ', ''). \
    #    split()
    #nodes = int(el_line[1])
    nodes = int(lines[3].replace('I=   ', ''))

    print('found %i nodes in the file\n'%nodes)
    
    for i in range(len(vars_slice)):
        vars_slice[i] = vars_slice[i].replace('"', '')
    vars_slice[-1] = vars_slice[-1].rstrip(' ')
    
    for i in range(len(vars_slice)):
        dicty[vars_slice[i]] = []
        for j in range(nodes):
            dicty[vars_slice[i]].append(float(lines[5+j+(nodes*i)]))
    return dicty


def my_read_yaml(finput):
    # Write yaml with updated values
    with open(finput, "r", encoding="utf-8") as f:
        dict = yaml.load(f, Loader=yaml.FullLoader)
    return dict

#-----------------------------------------------
#-----------------------------------------------

#returns only the items in pos and forces that are located on the blade that points in spanDir, at r>R0
#caution: span_b1 is the non-dimensional span location
def isolate_blade1(pos,forces,R0,R,spanDir):
    mask = pos[:,spanDir] >= R0
    pos_b1 = pos[mask,:]
    span_b1 = (pos_b1[:,spanDir].copy()-R0)/(R-R0)
    force_b1 = forces[mask,:]
    return span_b1,force_b1

#returns a matrix (Npos,4), each column is mask to identify the pts on the hub, and on the 3 blades
def mark_blades(pos, R0, spanDir, normDir):
    Nblade = 3 #hard
    Npts = np.shape(pos)[0]
    masks = np.zeros((Npts,Nblade+1),bool)

    r_loc = np.linalg.norm(pos,axis=1)

    chordDir = (3-spanDir-normDir)  #the direction normal to spanDir and nromDir

    #blade 1
    masks[:,1] =  pos[:,spanDir] >= R0
    #blade 2
    masks[:,2] =  (pos[:,spanDir] < 0) & (pos[:,chordDir] > 0) & (r_loc >= R0)
    #blade 3
    masks[:,3] =  (pos[:,spanDir] < 0) & (pos[:,chordDir] < 0) & (r_loc >= R0)
    #hub
    masks[:,0] = ~( masks[:,1] | masks[:,2] | masks[:,3]) #r_loc < R0
    

    return masks  

def RBF_lin(z,z0,rad) :
    return np.maximum(0.0 , 1.0 - abs(z-z0)/rad )


def aero_HiFi_2_Lofi(ref_aero_forces, output_aero_forces, rEL, FnEL, FtEL, R0, R, spanDir=1, normDir=0, fname_HFdistro=""):
    """ Scale the HiFi force distribution to match the LoFi loading distribution.

    :param ref_aero_forces: input file from ADflow (generated with `writeForceFile`)
    :param output_aero_forces: output file with scaled forces (same format)

    :param rEL: non dimensional radii [0.-1.] where the lofi load distribution are provided
    :type rEL: vect
    :param FnEL: lofi normal load distribution [N/m]
    :type FnEL: vect
    :param FtEL: lofi tangential load distribution [N/m]
    :type FtEL: vect
    :param R0: blade root cutoff radius [m]
    :param R: blade tip radius [m]

    :param spanDir: spanwise direction (0=x, 1=y, 2=z), defaults to 1
    :param normDir: rotor normal direction (0=x, 1=y, 2=z), defaults to 0
    :param fname_HFdistro: filename of the hifi load distribution [N/m] when using method=2, defaults to ""
    """

    pos,forces,conn,ncell = read_force_file(ref_aero_forces)
    span_b1,force_b1 = isolate_blade1(pos,forces,R0,R,spanDir)

    chordDir = (3-spanDir-normDir)  #the direction normal to spanDir and nromDir

    #------------ common definitions and processing ------------
    
    #define radial basis functions for every point in the distro
    myRBF = RBF_lin #choose a RBF

    #integrate the FnEL times each RBF: obtain a corresponding lofi force at every r of the distro
    n_interp = 1000 #need to reinterpolate lofi distro in case the number of stations is <= nnodes
    r_interp = np.linspace(0,1,n_interp)
    dr = r_interp[1] - r_interp[0]

    FnEL_interp = np.interp(r_interp, rEL, FnEL)
    FtEL_interp = np.interp(r_interp, rEL, FtEL)
        

    #define a uniform distribution of spanwise radii
    nnodes = 20
    rnodes = np.linspace(0,1,nnodes)
    rad = rnodes[1] - rnodes[0]

    FN_lofi_nodes = np.zeros(nnodes)
    FT_lofi_nodes = np.zeros(nnodes)
    for j in range(nnodes):
        FN_lofi_nodes[j] = np.trapz( myRBF(r_interp,rnodes[j],rad) * FnEL_interp ) * dr * (R-R0) #redimensionalize to get N
        FT_lofi_nodes[j] = np.trapz( myRBF(r_interp,rnodes[j],rad) * FtEL_interp ) * dr * (R-R0) #redimensionalize to get N

    #sum the hifi forces times each RBF:  obtain a corresponding hifi force at every r of the distro
    FN_hifi_nodes = np.zeros(nnodes)
    FT_hifi_nodes = np.zeros(nnodes)
    for j in range(nnodes):
        for i in range(len(span_b1)):
            FN_hifi_nodes[j] += myRBF(span_b1[i],rnodes[j],rad) * force_b1[i,normDir]
            FT_hifi_nodes[j] += myRBF(span_b1[i],rnodes[j],rad) * force_b1[i,chordDir]
    # force_b1 is in N/m^2 !!   
    # TODO: could determine chord from the pos, then would be able to have distro in N/m2 for the scaling without the need for lift file

    #------------ Determine scaling function ------------
    #------------ method 1 ------------
    if method == 1:
        #determine scaling as a function of r, that is the ratio between the forces above times the RBFs, sume over RBFs.
        scaling_nodes = FN_lofi_nodes/FN_hifi_nodes

        # print(FN_lofi_nodes)
        # print(FN_hifi_nodes)
        print(scaling_nodes)

        #define a scaling function object that I can evaluate at any r (spline interp?)
        # simply use the same RBF to interpolate
        # input z is non-dim
        def scale(z)  :
            out = 0
            for j in range(nnodes):
                out += myRBF(z,rnodes[j],rad) * scaling_nodes[j]

            return out

        fScaleR0 = scaling_nodes[0] #scaling factor from node 0 to hub

    #------------method 2------------
    elif method == 2: 

        #read hifi force distro 
        hf_distr = getLiftDistribution(fname_HFdistro)

        if spanDir==0:
            rHF = np.array(hf_distr['CoordinateX'][:])
        elif spanDir==1:
            rHF = np.array(hf_distr['CoordinateY'][:])
        else:
            rHF = np.array(hf_distr['CoordinateZ'][:])
        if normDir==0:
            FnHF = np.array(hf_distr['Fx'][:])
        elif normDir==1:
            FnHF = np.array(hf_distr['Fy'][:])
        else:
            FnHF = np.array(hf_distr['Fz'][:])
        if chordDir==0:
            FtHF = np.array(hf_distr['Fx'][:])
        elif chordDir==1:
            FtHF = np.array(hf_distr['Fy'][:])
        else:
            FtHF = np.array(hf_distr['Fz'][:])

        FnHF_interp = np.interp(r_interp, (rHF-R0)/(R-R0), FnHF)
        FtHF_interp = np.interp(r_interp, (rHF-R0)/(R-R0), FtHF)

        print(f"Max r in load distribution file: {max(rHF)/R} (caution if not exactly 1.0)!")
        print(f"Max r in force file: {max(span_b1)}")

        scaling = FnEL_interp / FnHF_interp

        #determine scaling as a function of r, define a function object that I can evaluate at any r (spline interp?)
        def scale(z)  :
            return np.interp(z, r_interp, scaling)
            
        fScaleR0 = scaling[0] #scaling factor from node 0 to hub

    else:
        raise RuntimeError("unkown method.")


    #------------Apply scaling to HF data------------
    
    #identify blade indices
    masks = mark_blades(pos, R0, spanDir, normDir)

    #scaling of hub forces:
    r_loc = np.linalg.norm(pos[masks[:,0],:],axis=1)
    fScale0 = 0.0 #linearly blend to 0 the forces on the nacelle: no need for them anyway
    for k in range(3):
        forces[masks[:,0],k] *= ( r_loc/R0 * fScaleR0 + (R0-r_loc)/R0 * fScale0)
    
    #scaling of blade1 forces:
    r_loc = (pos[masks[:,1],spanDir] - R0)/(R-R0)
    for k in range(3):
        forces[masks[:,1],k] *= scale(r_loc)

    #scaling of blade2 forces:
    r_loc = (pos[masks[:,2],spanDir] * np.cos(2*np.pi/3) + pos[masks[:,2],chordDir] * np.sin(2*np.pi/3)  - R0)/(R-R0)
    for k in range(3):
        forces[masks[:,2],k] *= scale(r_loc)

    #scaling of blade3 forces:
    r_loc = (pos[masks[:,3],spanDir] * np.cos(4*np.pi/3) + pos[masks[:,3],chordDir] * np.sin(4*np.pi/3)  - R0)/(R-R0)
    for k in range(3):
        forces[masks[:,3],k] *= scale(r_loc)

    
    #------------ Export ------------

    #redo node approach on scaled distribution on hifi
    FN_hifi_nodes_scaled = np.zeros(nnodes)
    FT_hifi_nodes_scaled = np.zeros(nnodes)
    span_b1,force_b1 = isolate_blade1(pos,forces,R0,R,spanDir)
    for j in range(nnodes):
        for i in range(len(span_b1)):
            FN_hifi_nodes_scaled[j] += myRBF(span_b1[i],rnodes[j],rad) * force_b1[i,normDir]
            FT_hifi_nodes_scaled[j] += myRBF(span_b1[i],rnodes[j],rad) * force_b1[i,chordDir]


    if debug:
        # fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
        # plt.plot(rnodes,FN_lofi_nodes)
        # plt.plot(rnodes,FN_hifi_nodes)
        # # plt.plot(rnodes,FN_hifi_nodes*scaling_nodes,':') #theoretical 
        # plt.plot(rnodes,FN_hifi_nodes_scaled,'--')
        # plt.ylabel("Fn [N]")

        # fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
        # plt.plot(rnodes,FT_lofi_nodes)
        # plt.plot(rnodes,FT_hifi_nodes)
        # # plt.plot(rnodes,FT_hifi_nodes*scaling_nodes,':') #theoretical 
        # plt.plot(rnodes,FT_hifi_nodes_scaled,'--')
        # plt.ylabel("Ft [N]")

        if method == 2:
            fig3, ax3 = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
            plt.plot(r_interp,FnEL_interp)
            plt.plot(r_interp,FnHF_interp)
            plt.plot(r_interp,FnHF_interp*scaling,'--')
            plt.ylabel("Fn [N/m]")
            plt.savefig("Fn.png")

            fig4, ax4 = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
            plt.plot(r_interp,FtEL_interp)
            plt.plot(r_interp,FtHF_interp)
            plt.plot(r_interp,FtHF_interp*scaling,'--')
            plt.ylabel("Ft [N/m]")
            plt.savefig("Ft.png")

            fig5, ax5 = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
            plt.plot(r_interp,scaling,'--')
            plt.ylabel("scaling")
            plt.savefig("scaling.png")


        # fig = plt.figure()
        # ax = plt.axes(projection='3d')

        # SCL = .1 * max( np.linalg.norm( pos , axis=1) ) / max( np.linalg.norm( forces , axis=1) ) 
        # for k in range( np.shape(forces)[0] ):
        #     ax.plot3D(pos[k,0] + SCL * np.array([0, forces[k,0]]), 
        #               pos[k,1] + SCL * np.array([0, forces[k,1]]), 
        #               pos[k,2] + SCL * np.array([0, forces[k,2]]), 'blue')

        plt.show()
        

    exportToXdmf("scaled.xmf",pos,forces)
    write_force_file(output_aero_forces,pos,forces,conn,ncell)


#-----------------------------------------------
# ================================================

def exportToXdmf(fname,pos,forces):

    npts = np.shape(pos)[0]
    with open(fname,'w') as f:
        f.write("<?xml version=\"1.0\" ?>\n")
        f.write("<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n")
        f.write("<Xdmf xmlns:xi=\"http://www.w3.org/2003/XInclude\" Version=\"2.2\">\n")

        f.write("<Domain>\n")
        f.write("    <Grid Name=\"CellTime\" GridType=\"Collection\" CollectionType=\"Temporal\">\n")
        f.write("    <Grid GridType=\"Uniform\">\n")
        f.write("          <Time Value=\"0.00\" />\n")
        f.write("          <Topology TopologyType=\"Polyvertex\" NodesPerElement=\"2\">\n")
        f.write("          </Topology>\n")
        f.write("          <Geometry GeometryType=\"XYZ\">\n")
        f.write(f"                <DataItem DataType=\"Float\" Dimensions=\"{npts} 3\" Format=\"XML\">\n")
        for k in range(npts):
            f.write("             %6.5f %6.5f %6.5f\n"%(pos[k,0],pos[k,1],pos[k,2]))
        f.write("                  </DataItem>\n")
        f.write("            </Geometry>\n")
            # <Attribute AttributeType=\"Scalar\" Center=\"Node\"  Name=\"leuk_type\">
            #       <DataItem DataType=\"Int\" Dimensions=\"30 1\"  Format=\"HDF\">
            #             TestData.h5:/iter00000000/cells/type
            #       </DataItem>
            # </Attribute>
        f.write("            <Attribute AttributeType=\"Vector\" Center=\"Node\"  Name=\"force\">\n")
        f.write(f"                  <DataItem DataType=\"Float\" Dimensions=\"{npts} 3\" Precision=\"4\"  Format=\"XML\">\n")
        for k in range(npts):
            f.write("             %6.5f %6.5f %6.5f\n"%(forces[k,0],forces[k,1],forces[k,2]))
        f.write("                  </DataItem>\n")
        f.write("            </Attribute>\n")
        f.write("      </Grid>\n")
        f.write("      </Grid>\n")
        f.write("  </Domain>\n")
        f.write("</Xdmf>\n")



def write_force_file(fname,pos,forces,conn,ncell):

    npts = np.shape(pos)[0]
    with open(fname,'w') as f:
        f.write("%i %i\n"%(npts,ncell))
        for k in range(npts):
            f.write("   %6.5f %6.5f %6.5f %6.5f %6.5f %6.5f\n"%(pos[k,0],pos[k,1],pos[k,2],forces[k,0],forces[k,1],forces[k,2]))
        for el in conn:
            f.write("%s"%(el))

# V==============================================v

if __name__=='__main__':

    from OTCDparser import OFparse

    FastFile = "inputs/DTU_10MW_V8_TSR781.out"

    #EXTREME/FATIGUE LOADS
    loadFolder = "results-IEC1.1_5vels_120s_0Glob"
    loadFolder = "results-IEC1.3_5vels_120s_0Glob" #DEL computed with ETW... not recommended because too much, should always use NTW for fatigue (see DLC1.2)
    loadFolder = "results-IEC1.1-IEC1.3_5vels_120s_0Glob_chi2" #DEL computed with ETW... not recommended because too much, should always use NTW for fatigue (see DLC1.2)
    loadFolder = "results-IEC1.1_5vels_120s_0Glob_feq10"
    loadFolder = "results-IEC1.1_5vels_120s_0Glob_neq1"
    loadFile = f"../07_test_iterateDEL/{loadFolder}/aggregatedEqLoads.yaml"

    #NOMINAL LOADS
    loadFolder = "Madsen2019_10_forWEIS_isotropic"
    loadFile = f"../09_compare_struct/{loadFolder}/nominalLoads.yaml"

    R0 = 2.8
    R = 89.166
    meshLevel = "L2" #note that the conversion of aero load to structural load can handle any load level.

    ref_HiFi_forceFile = f"inputs/force_allwalls_{meshLevel}_0.txt"
    ref_HiFi_liftFile = f"inputs/Analysis_DTU10MW_V8_TSR781_000_lift.dat" #MUST BE ADAPTED DEPENDING ON MESH LEVEL

    suff = ""
    if loFiSource ==1:
        suff = "DUMMY"
        dummy_r    = [0,.5,.99,1.] 
        dummy_FnEL = np.array([1.,1.,1.,0.])*1e2
        dummy_FtEL = np.array([1.,1.,1.,0.])*1e1
        
        aero_HiFi_2_Lofi(ref_HiFi_forceFile,f"force_allwalls_{meshLevel}_{suff}.txt",dummy_r, dummy_FnEL, dummy_FtEL, R0, R, fname_HFdistro=ref_HiFi_liftFile)
    elif loFiSource ==2:
        suff = "STEADY_REF"
        dummy_r = np.array([0.000000e+00, 2.643018e+00, 5.379770e+00, 8.202738e+00, 1.110314e+01, 1.407102e+01, 1.709533e+01, 2.016412e+01, 2.326467e+01, 2.638370e+01, 2.950764e+01, 3.262279e+01, 3.571564e+01, 3.877302e+01, 4.178239e+01, 4.473200e+01, 4.761106e+01, 5.040992e+01, 5.312010e+01, 5.573444e+01, 5.824705e+01, 6.065336e+01, 6.295010e+01, 6.513516e+01, 6.720760e+01, 6.916749e+01, 7.101585e+01, 7.275451e+01, 7.438597e+01, 7.591335e+01, 7.734022e+01, 7.867052e+01, 7.990848e+01, 8.105852e+01, 8.212516e+01, 8.311296e+01, 8.402649e+01, 8.487025e+01, 8.564866e+01, 8.636600e+01])/(R-R0)
        _, _, _, dummy_FnEL, dummy_FtEL = OFparse(FastFile, nodeR=dummy_r)

        dummy_FnEL[-1] = 0 #make sure the last loading evaluated at r=R in this case is 0

        aero_HiFi_2_Lofi(ref_HiFi_forceFile,f"force_allwalls_{meshLevel}_{suff}.txt",dummy_r, dummy_FnEL, dummy_FtEL, R0, R, fname_HFdistro=ref_HiFi_liftFile)
    else:
        dict = my_read_yaml(loadFile)
        
        if not os.path.isdir("./" + loadFolder):
            os.makedirs("./" + loadFolder)

        suffs = ["DEL","extreme","nominal"]

        for suff in suffs:
            if suff in dict.keys():
                r = dict[suff]["grid_nd"]
                FnEL = dict[suff]["Fn"]
                FtEL = dict[suff]["Ft"]
                FnEL[-1] = 0 #make sure the last loading evaluated at r=R in this case is 0

                aero_HiFi_2_Lofi(ref_HiFi_forceFile,f"./{loadFolder}/force_allwalls_{meshLevel}_{suff}.txt",r, FnEL, FtEL, R0, R, fname_HFdistro=ref_HiFi_liftFile)
                for f in fl:
                    if os.path.isfile(f):
                        os.system(f"mv {f} {loadFolder}/{f.split('.')[0]}_{suff}.{f.split('.')[1]}")

   
    
    