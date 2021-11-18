import os

# import sys, shutil
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import extrapolate_utils as exut

folder = "results"
saveExtrNpy = "extrmDistro.npz"


labs = ["Fn [N/m]","Ft [N/m]","MLx [kNm]","MLy [kNm]","FLz [kN]"]

badThingsHappen = 50 #reccurring period [years]

iplt = [5,15,25] #spanwise stations to plot
# iplt = [23] #75%

reProcess = True
barplots = False
pltSize = (10, 5)
# pltSize = (6, 3)


mydistr = ["weibull_min","weibull_min","norm","norm","norm"]
mydistr = ["chi2","chi2","chi2","chi2","chi2"]
# mydistr = ["chi2","chi2","twiceMaxForced","norm","norm"] #chi2 curve fitting may lead to oscillations in the output loading
# mydistr = ["norm","norm","norm","norm","norm"] #safer from a numerical perspective
# mydistr = ["gumbel_r","gumbel_r","gumbel_r","gumbel_r","gumbel_r",]
# mydistr = ["weibull_min","weibull_min","weibull_min","weibull_min","weibull_min"]
# mydistr = ["normForced","normForced","normForced","normForced","normForced"]

truncThr = None #We discard all data on the left of the distro `avg + truncThr * std` for the fit. If None, keep everything.
truncThr = 0.5

# # new recommended setup:
mydistr = ["norm","norm","twiceMaxForced","norm","norm"] 
truncThr = [0.5,1.0,None,0.5,0.5]

# ========================================

f= np.load(folder + os.sep + saveExtrNpy)

rng = f["rng"]
nbins = f["nbins"]
EXTR_life_B1 = f["EXTR_life_B1"]
EXTR_distr_p = f["EXTR_distr_p"]
EXTR_distro_B1 = f["EXTR_distro_B1"]
distr = f["distr"]
dt = f["dt"]

IEC_50yr_prob = 1. - dt / (badThingsHappen*3600*24*365)


if reProcess:
    distr = mydistr
    # if extremeExtrapMeth ==1:
    #     #assumes only normal
    #     EXTR_life_B1, EXTR_distr_p = extrapolate_extremeLoads_hist(rng, EXTR_distro_B1,IEC_50yr_prob)
    # elif extremeExtrapMeth ==2:
    #     EXTR_life_B1, EXTR_distr_p = extrapolate_extremeLoads(EXTR_data_B1, distr, IEC_50yr_prob)
    # elif extremeExtrapMeth ==3:
    EXTR_life_B1, EXTR_distr_p = exut.extrapolate_extremeLoads_curveFit(rng, EXTR_distro_B1, distr, IEC_50yr_prob, truncThr=truncThr)




for k in range(5):
    f1,ax1 = plt.subplots(nrows=1, ncols=1, figsize=pltSize)
    f2,ax2 = plt.subplots(nrows=1, ncols=1, figsize=pltSize)

    ax1.set_ylabel("probability density")
    ax2.set_ylabel("probability of exceedance")

    print(EXTR_distr_p[5,k,:])
    print(EXTR_distr_p[15,k,:])
    print(EXTR_distr_p[25,k,:])

    for i in iplt:
        stp = (rng[k][1]-rng[k][0])/(nbins)
        xbn = np.arange(rng[k][0]+stp/2.,rng[k][1],stp) #(bns[:-1] + bns[1:])/2.
        dx = (rng[k][1]-rng[k][0])/(nbins)
        
        
        ax1.set_xlabel(labs[k])
        
        if barplots:
            ss1 = ax1.bar(xbn,EXTR_distro_B1[i,k,:] ,width=0.8*dx)
            c1 = ss1[0].get_facecolor()
        else:
            ss1 = ax1.plot(xbn,EXTR_distro_B1[i,k,:] )
            c1 = ss1[0].get_color()
        ax1.plot(EXTR_life_B1[i,k] , 0, 'x' , color=c1)
        
        
        ax2.set_yscale('log')
        ax2.set_xlabel(labs[k])
        ax2.set_ylim([ (1.-IEC_50yr_prob)/2. , 2.])                

        dsf1= 1.-np.cumsum(EXTR_distro_B1[i,k,:] )*dx 
        dsf1[(dsf1>=1.-1E-16) | (dsf1<=1E-16)] = np.nan
        ax2.plot(xbn,dsf1)
        ax2.plot(EXTR_life_B1[i,k] , 1.-IEC_50yr_prob, 'x' , color=c1)
        
        
        x_upper = np.min([EXTR_life_B1[i,k],rng[k][1]*100])
        xx = np.arange(rng[k][0]+dx/2.,x_upper,dx)


        if "twiceMaxForced" in distr[k]:
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
        # ax2.plot([xx[0],xx[-1]],[1.-IEC_50yr_prob,1.-IEC_50yr_prob],'-k',linewidth=0.5 )
    f1.tight_layout()
    f2.tight_layout()
    f1.savefig(f"{folder}/figs/fit_{labs[k].split(' ')[0]}_{distr[k]}.eps")
    f2.savefig(f"{folder}/figs/fit_sf_{labs[k].split(' ')[0]}_{distr[k]}.eps")

plt.show()

nx=np.size(EXTR_life_B1,axis=0)
locs = np.linspace(0.,1.,nx)

for k in range(5):
    plt.subplots(nrows=1, ncols=1, figsize=pltSize)
    plt.plot(locs,EXTR_life_B1[:,k], label="EXTR")
    
    plt.ylabel(labs[k])
    plt.xlabel("r/R")
    # plt.legend()

    plt.tight_layout()
    plt.savefig(f"{folder}/figs/{labs[k].split(' ')[0]}_{distr[k]}.eps")
plt.show()