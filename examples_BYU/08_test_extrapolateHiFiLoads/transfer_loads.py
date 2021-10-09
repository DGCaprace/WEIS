import os
# import sys, shutil
import ast
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import io

debug = True
fakeHiFi = False

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
        npts = np.int(buff[0])
        ncell = np.int(buff[1])

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

#directions: 0=x, 1=y, 2=z
def aero_HiFi_2_Lofi(ref_aero_forces, output_aero_forces, rEL, FnEL, FtEL, R0, R, spanDir=1, normDir=0):

    pos,forces,conn,ncell = read_force_file(ref_aero_forces)
    span_b1,force_b1 = isolate_blade1(pos,forces,R0,R,spanDir)

    chordDir = (3-spanDir-normDir)  #the direction normal to spanDir and nromDir

    #method 1
    #define a uniform distribution of spanwise radii
    nnodes = 20
    rnodes = np.linspace(0,1,nnodes)
    rad = rnodes[1] - rnodes[0]

    #define radial basis functions for every point in the distro
    myRBF = RBF_lin #choose a RBF

    #integrate the FnEL times each RBF: obtain a corresponding lofi force at every r of the distro
    n_interp = 1000 #need to reinterpolate lofi distro in case the number of stations is <= nnodes
    r_interp = np.linspace(0,1,n_interp)
    dr = r_interp[1] - r_interp[0]
    FN_lofi_nodes = np.zeros(nnodes)
    FT_lofi_nodes = np.zeros(nnodes)
    FnEL_interp = np.interp(r_interp, rEL, FnEL)
    FtEL_interp = np.interp(r_interp, rEL, FtEL)
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


    #identify blade indices
    masks = mark_blades(pos, R0, spanDir, normDir)

    #scaling of hub forces:
    r_loc = np.linalg.norm(pos[masks[:,0],:],axis=1)
    for k in range(3):
        forces[masks[:,0],k] *= ( r_loc/R0 * scaling_nodes[0] + (R0-r_loc)/R0 * 1.)
    
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

    
    if debug:
        fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
        plt.plot(rnodes,FN_lofi_nodes)
        plt.plot(rnodes,FN_hifi_nodes)
        plt.ylabel("Fn [N]")

        fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
        plt.plot(rnodes,FT_lofi_nodes)
        plt.plot(rnodes,FT_hifi_nodes)
        plt.plot(rnodes,FT_hifi_nodes*scaling_nodes,'--')
        plt.ylabel("Ft [N]")

        fig = plt.figure()
        ax = plt.axes(projection='3d')

        print(min( np.linalg.norm( pos , axis=1) ))
        SCL = .1 * max( np.linalg.norm( pos , axis=1) ) / max( np.linalg.norm( forces , axis=1) ) 
        for k in range( np.shape(forces)[0] ):
            ax.plot3D(pos[k,0] + SCL * np.array([0, forces[k,0]]), 
                      pos[k,1] + SCL * np.array([0, forces[k,1]]), 
                      pos[k,2] + SCL * np.array([0, forces[k,2]]), 'blue')

        # print(pos)
        # print(forces)

        plt.show()
        


    #method 2
    #read hifi force distro 
    #determine scaling as a function of r, define a function object that I can evaluate at any r (spline interp?)
    #as a check, apply scaling to Ft hifi and lofi distro, and compare those
    #apply scaling to force file

    if debug:
        exportToXdmf("scaled.xmf",pos,forces)

    write_force_file(output_aero_forces,pos,forces,conn,ncell)



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
        f.write("%s"%(conn))

# V==============================================v

if __name__=='__main__':

    dummy_r    = [0,.5,1.] 
    dummy_FnEL = np.array([1.,1.,1.])*1e2
    dummy_FtEL = np.array([1.,1.,1.])*1e1

    R0 = 2.8
    R = 89.166

    aero_HiFi_2_Lofi("force_allwalls_L3.txt","force_allwalls_L3_RESCALED.txt",dummy_r, dummy_FnEL, dummy_FtEL, R0, R)