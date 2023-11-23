import os

# import sys, shutil
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import extrapolate_utils as exut

folder = "results"
folder = "results-IEC1.1-IEC1.3_12vels_600s"
saveExtrNpy = "extrmDistro.npz"
dlc_label = '1.3' #label of the dlc in the npz that we want to process

labs = ["Fn [N/m]","Ft [N/m]",
        "MLx [kNm]","MLy [kNm]","FLz [kN]",
        "StrainU [-]","StrainL [-]","StrainTE [-]",]
legs = [r"$F_n \, [N/m]$",r"$F_t \, [N/m]$",
        r"$ML_x \, [kNm]$",r"$ML_y \, [kNm]$",r"$FL_z \, [kN]$",
        r"$\epsilon_U \, [-]$",r"$\epsilon_L \, [-]$",r"$\epsilon_{TE} \, [-]$",]

Textrm = 50 #return period of extreme event [years]

iplt = [15] #spanwise stations to plot
# iplt = [23] #75%

reProcess = True
reNormalize = True #make sure that all distributions sum to 1.00
logfit = True   #proceed to fitting the log of the survival function. If false, we fit the pdf. Recommend truncThr=None if logfit=True
barplots = True
pltSize = (10, 5)
pltSize = (6, 3)

fs = 14
ls = 12

#CAUTION: we assume extremeExtrapMeth=3

mydistr = [] #use the one coming from the file 
mytruncThr = [] #We discard all data on the left of the distro `avg + truncThr * std` for the fit. If None, keep everything.

mydistr = ["norm","norm","norm","norm","norm","maxForced","norm","norm"] 
mytruncThr = [0.5,1.0,1.0,0.5,0.5,None,1.0,0.5] 
#[norm and 1.0] seems to be working well for bimodal distributions

k_SU = 5 #revert the upper spar strain when computing the tail of the distribution so that it's on the right (>0)

# ========================================

killUnder = 1E-14 #remove all values in the experimental distribution under this threshold (numerical noise)

f= np.load(folder + os.sep + saveExtrNpy, allow_pickle=True)


# #For new files:
# dlcs = f["iec_extr"]  
# dlc = dlcs[0]#assuming the dlc we want to look at is at index 0
# EXTR_distro_B1 = dlc["binned_loads"]
# EXTR_life_B1 = dlc["extr_loads"]
# EXTR_distr_p = dlc["extr_params"]
# EXTR_distro_B1 = EXTR_distro_B1[:,:,:,0] #assume there is only 1 timeserie, as a result of aggregation

#After update to WEIS 1.0:
print(f.files)
dlcs = f["DLCs_extr"].item() #extracting the dictionary
dlc = dlcs[dlc_label]#assuming the dlc we want to look at is at index 0
EXTR_distro_B1 = dlc["binned_loads"]
EXTR_life_B1 = dlc["extr_loads"][0]
EXTR_distr_p = dlc["extr_params"][0]
EXTR_distro_B1 = EXTR_distro_B1[:,:,:,0]

if len(mydistr)==0:
    distr = f["distr"]
else: 
    distr = mydistr
    
if not( mytruncThr is None or len(mytruncThr)!=0 ):
    if "truncThr" in f:
        truncThr = f["truncThr"]
    else:
        print("WARNING: truncThr not in the npz file. I am assuming =None.")
    truncThr = None
else:
    truncThr = mytruncThr

dt = f["dt"]

rng = f["rng"] #can't change range and bins without recomputing the whole binning, i.e. rerunning the complete postpro
nbins = f["nbins"]
if "rng_mod" in f:
    rng_mod = f["rng_mod"]
else:
    rng_mod = np.ones((EXTR_distro_B1.shape[1],EXTR_distro_B1.shape[0]))

IEC_50yr_prob = 1. - dt / (Textrm*3600*24*365)


# CORRECT FOR NUMERICAL ERROR: make sure that the EXTR_distro sum to 1.00
n1 = np.shape(EXTR_distro_B1)[0]
n2 = np.shape(EXTR_distro_B1)[1]

for k in range(n2):
    if reNormalize:
        for i in range(n1):
            stp = (rng[k][1]-rng[k][0])/(nbins) * rng_mod[k,i]
            EXTR_distro_B1[i,k,:] /= np.sum(EXTR_distro_B1[i,k,:]*stp)
    else:
        for i in range(n1):
            stp = (rng[k][1]-rng[k][0])/(nbins) * rng_mod[k,i]
            dsum = np.sum(EXTR_distro_B1[i,k,:]*stp)
            if abs(dsum-1.0) > 1e-5 and not reNormalize:
                print(f'WARNING: member [{i},{k},:] of the distro matrix does not sum to 1 ({dsum}). Recommend turning reNormalize=True.')


if reProcess:
    distr = mydistr

    #CHANGE SIGN ASSUNING SUCTION SIDE WILL BE COMPRESSED
    EXTR_distro_B1[:,k_SU,:] = -EXTR_distro_B1[:,k_SU,:]                                
                                    
    #TODO: instead of copy pasting from driver, should do this a little better
    # if extremeExtrapMeth ==1:
    #     #assumes only normal
    #     EXTR_life_B1, EXTR_distr_p = extrapolate_extremeLoads_hist(rng, EXTR_distro_B1,IEC_50yr_prob)
    # elif extremeExtrapMeth ==2:
    #     EXTR_life_B1, EXTR_distr_p = extrapolate_extremeLoads(EXTR_data_B1, distr, IEC_50yr_prob)
    # elif extremeExtrapMeth ==3:
    EXTR_life_B1, EXTR_distr_p = exut.extrapolate_extremeLoads_curveFit(rng, EXTR_distro_B1, distr, IEC_50yr_prob, truncThr=truncThr, logfit=logfit, killUnder=killUnder, rng_mod=rng_mod)

    #REVERTING
    EXTR_life_B1[:,k_SU] = -EXTR_life_B1[:,k_SU] 
    EXTR_distro_B1[:,k_SU,:] = -EXTR_distro_B1[:,k_SU,:] 


for k in range(8):
    f1,ax1 = plt.subplots(nrows=1, ncols=1, figsize=pltSize)
    f2,ax2 = plt.subplots(nrows=1, ncols=1, figsize=pltSize)

    ax1.tick_params(labelsize=ls)
    ax2.tick_params(labelsize=ls)

    ax1.set_ylabel("probability density",fontsize=fs)
    ax2.set_ylabel("probability of exceedance",fontsize=fs)

    # print(EXTR_distr_p[5,k,:])
    # print(EXTR_distr_p[15,k,:])
    # print(EXTR_distr_p[25,k,:])

    for i in iplt:
        stp = (rng[k][1]-rng[k][0])/(nbins)
        xbn = np.arange(rng[k][0]+stp/2.,rng[k][1],stp) #(bns[:-1] + bns[1:])/2.
        stp *= rng_mod[k,i]
        xbn *= rng_mod[k,i]
        dx = stp
        
        
        ax1.set_xlabel(legs[k],fontsize=fs)
        
        if barplots:
            ss1 = ax1.bar(xbn,EXTR_distro_B1[i,k,:] ,width=0.8*dx)
            c1 = ss1[0].get_facecolor()
        else:
            ss1 = ax1.plot(xbn,EXTR_distro_B1[i,k,:] )
            c1 = ss1[0].get_color()
        ax1.plot(EXTR_life_B1[i,k] , 0, 'x' , color=c1)
        
        
        ax2.set_yscale('log')
        ax2.set_xlabel(legs[k],fontsize=fs)
        ax2.set_ylim([ (1.-IEC_50yr_prob)/2. , 2.])                

        dsf1= 1.-np.cumsum(EXTR_distro_B1[i,k,:] )*dx 
        dsf1[(dsf1>=1.-killUnder) | (dsf1<=killUnder)] = np.nan
        ax2.plot(xbn,dsf1)
        ax2.plot(EXTR_life_B1[i,k] , 1.-IEC_50yr_prob, 'x' , color=c1)
        

        if "twiceMaxForced" in distr[k]:
            pass
        if "maxForced" in distr[k]:
            pass
        elif "normForced" in distr[k]:
            ax1.plot(xx, stats.norm.pdf(xx, loc = EXTR_distr_p[i,k,0], scale = EXTR_distr_p[i,k,1]),'--', alpha=0.6 , color='black')
            ax2.plot(xx, stats.norm.sf(xx, loc = EXTR_distr_p[i,k,0], scale = EXTR_distr_p[i,k,1]),'--', alpha=0.6 , color='black')
        elif "norm" in distr[k] or "gumbel" in distr[k]: #2params models
            this = getattr(stats,distr[k])
            ax1.plot(xx, this.pdf(xx, loc = EXTR_distr_p[i,k,0], scale = EXTR_distr_p[i,k,1]),'--', alpha=0.6 , color='black')
            ax2.plot(xx, this.sf(xx, loc = EXTR_distr_p[i,k,0], scale = EXTR_distr_p[i,k,1]),'--', alpha=0.6 , color='black')
        else: #3params models
            this = getattr(stats,distr[k])
            ax1.plot(xx, this.pdf(xx, EXTR_distr_p[i,k,0], loc = EXTR_distr_p[i,k,1], scale = EXTR_distr_p[i,k,2]),'--', alpha=0.6 , color='black')
            ax2.plot(xx, this.sf(xx, EXTR_distr_p[i,k,0], loc = EXTR_distr_p[i,k,1], scale = EXTR_distr_p[i,k,2]),'--', alpha=0.6 , color='black')

        if not truncThr is None and not truncThr[k] is None:
            avg = np.sum( EXTR_distro_B1[i,k,:] * xbn ) / np.sum(EXTR_distro_B1[i,k,:])
            std = np.sqrt( np.sum( EXTR_distro_B1[i,k,:] * xbn**2 ) / np.sum(EXTR_distro_B1[i,k,:]) - avg**2 )
            print(f"AVG and STD:",avg,std)
            ax1.plot([avg+truncThr[k]*std,]*2, [0.,2*np.max(EXTR_distro_B1[i,k,:])],'.-', color=c1)

        
        # ax2.plot([xx[0],xx[-1]],[1.-IEC_50yr_prob,1.-IEC_50yr_prob],'-k',linewidth=0.5 )
    f1.tight_layout()
    f2.tight_layout()
    if not os.path.isdir(f"{folder}/figs"):
            os.makedirs(f"{folder}/figs")
    f1.savefig(f"{folder}/figs/fit_{labs[k].split(' ')[0]}_{distr[k]}.eps")
    f2.savefig(f"{folder}/figs/fit_sf_{labs[k].split(' ')[0]}_{distr[k]}.eps")

plt.show()

pltSize = (6, 3)
fs = 20
ls = 15

nx=np.size(EXTR_life_B1,axis=0)
locs = np.linspace(0.,1.,nx)

for k in range(5):
    f1,ax1 = plt.subplots(nrows=1, ncols=1, figsize=pltSize)
    ax1.tick_params(labelsize=ls)

    plt.plot(locs,EXTR_life_B1[:,k], label="EXTR")
    
    plt.ylabel(labs[k],fontsize=fs)
    plt.xlabel(r"$r/R$",fontsize=fs)
    # plt.legend()

    plt.tight_layout()
    plt.savefig(f"{folder}/figs/{labs[k].split(' ')[0]}_{distr[k]}.eps")
plt.show()