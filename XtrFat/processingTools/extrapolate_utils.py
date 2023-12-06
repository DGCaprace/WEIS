import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

def __choose_which_side(mymin,mymax):
    side = np.sign(abs(mymax)-abs(mymin))
    extr = mymax if side>0 else mymin  #the max of the absolute value, but keep the sign
    return extr, side

def __compute_avg_std(mat,x):
    avg = np.sum( mat * x ) / np.sum(mat)
    std = np.sqrt( np.sum( mat * x**2 ) / np.sum(mat) - avg**2 )
    return avg, std

def __list_to_val(truncThr,k):
    if truncThr is None or (hasattr(truncThr,"__len__") and ( (len(truncThr)==0) or (truncThr[k] is None) )):
        threshold = None
    elif hasattr(truncThr,"__len__"):
        threshold = truncThr[k]
    else:
        threshold = truncThr
    return threshold

def __truncate_distro(x, mat, truncThr, avg, std, keepAtLeast, side, distr):
    #compute the range over which the fit must be done, if user asked to reduce the dataset
    if truncThr is None:
        #don't trash or truncate anything
        spn = np.array(range(len(mat)))
        if distr in ['norm','gumbel_l','gumbel_r']:
            p0 = [avg,std] #initial search point
        else:
            p0=[20,-avg/2,std/2]  #empirical way of determining initial condition, works ok with chi2
    else:
        if side>0:
            spn = np.where(x>=avg+truncThr*std)[0]
            if distr in ['norm','gumbel_l','gumbel_r']:
                p0 = [avg+truncThr*std,std/truncThr] #move initial search point towards the tail
            else:
                p0=[5,avg,std/2]
        else:
            spn = np.where(x<=avg-truncThr*std)[0]
            if distr in ['norm','gumbel_l','gumbel_r']:
                p0 = [avg-truncThr*std,std/truncThr] #move initial search point towards the tail
            else:
                p0=[5,avg,std/2] #NOTE: not quite sure here
    if keepAtLeast>0:
        if len(spn) == 0:
            print("Warning: empty set. Reintroduce the whole dataset.")
            spn = range(len(mat))
        elif len(spn)<keepAtLeast:
            spn = range(spn[-1]-keepAtLeast,spn[-1]+1)
    
    return spn, p0

def extrapolate_extremeLoads_hist(rng,mat,extr_prob):
    nbins = np.shape(mat)[2]
    n1 = np.shape(mat)[0]
    n2 = np.shape(mat)[1]

    p = np.nan*np.zeros((n1,n2,2))

    for k in range(n2):
        stp = (rng[k][1]-rng[k][0])/(nbins)
        x = np.arange(rng[k][0]+stp/2.,rng[k][1],stp)
        for i in range(n1):
            p[i,k,0] = np.sum( mat[i,k,:] * x ) / np.sum(mat[i,k,:])
            p[i,k,1] = np.sqrt( np.sum( mat[i,k,:] * x**2 ) / np.sum(mat[i,k,:]) - p[i,k,0] )
   
    n_std_extreme = stats.norm.ppf(extr_prob)
    EXTR_life_B1 = p[:,:,0] + n_std_extreme * p[:,:,1]

    side = 1 #TODO

    return EXTR_life_B1, p, side
    

def extrapolate_extremeLoads(mat, distr_list, extr_prob):
    n1 = np.shape(mat)[0]
    n2 = np.shape(mat)[1]

    extr = np.zeros((n1,n2))

    p = np.nan*np.zeros((n1,n2,3)) #not very general ...

    for k in range(n2):
        distr = getattr(stats,distr_list[k])
        if 'norm' in distr_list[k]:
            for i in range(n1):
                params = distr.fit(mat[i,k,:])
                extr[i,k] = distr.ppf(extr_prob, loc = params[0], scale = params[1])
                # myppf = distr.ppf(extr_prob)
                # extr[i,k] = params[0] + myppf * params[1]
                p[i,k,0] = params[0] #can I do something better than this?
                p[i,k,1] = params[1] #can I do something better than this?
        else:
            #not sure the implementation with loc and scale is super general
            for i in range(n1):
                params = distr.fit(mat[i,k,:])
                extr[i,k] = distr.ppf(extr_prob, params[0], loc=params[1], scale=params[2])
                p[i,k,0] = params[0] #can I do something better than this?
                p[i,k,1] = params[1] #can I do something better than this?
                p[i,k,2] = params[2] #can I do something better than this?
            
    side = np.ones((n1,n2)) #TODO

    return extr, p, side

"""
Tool to extrapolate the extreme loads based on a fitting of the pdf.
The func will try to fit both tails of the distro specified by the histogram passed in mat,
for a number of K quantities over I span locations, which are distributed over L
bins within the range rng. For each K quantity, the fitted distribution can be different,
and is specified by K distr_list. Eventually, the function uses the fits to extrapolate
the value corresponding to the residual probability extr_prob.

The fitting can use some tricks. The binned data can be restricted to some extent so
that the fitting occurs only on part of the histogram. `truncThr` allows to provide
a truncation threshold to isolate the tail: e.g., for the right tail, we discard all
the bins located in `x` such that `x < avg + threshol * std`, where avg is the average 
of the historgram and std is the standard dev. Another way of discarding bins is 
by looking at the corresponding probability. One may want to kill bins under a certain
probability (when it becomes close to machine precision) with `killUnder`.
The last trick is to perform the fit in log space. This is super powerful to emphasize the
tail of the distribution and improve the match there.

Inputs:
- rng : a matrix of size K,2 of floats - the bounds of the 
- mat : a matrix of size I,K,L of floats - the normalized probability of getting quantity K at spanwise station I in bin L (normalization should be done before!)
- distr_list : a list of size L of str - the name of the distribution to be fitted for quantity K, among those available in the scipy.stats package. 
        Currently supported are "norm","gumbel_l/r","chi","chi2","weibull_min/max". But technically any can be used as long as it's a 2 or 3 parameter distro.
- extr_prob : 
- truncThr=None : None or a float or a list of size K of None and floats -  The threshold for truncation of data. Only the data in bins such that x_bin > avg + truncThr*std
        are considered for the right tail (x_bin < avg - truncThr*std for the left tail)
- keepAtLeast=15 : Int - the minimum number of bins to keep after the threshold-based truncation. There should still be bins in the historgram to fit!
- logfit=False : Bool - If True, perform the fit on the log of the probability. Otherwise, perform it on the probability itself.
- killUnder=1e-14 : float - allows to kill all the bins which probability is under killUnder. This value should remain close to machine precision, but is necessary to
        discard bins with probability of 0.0, which would otherwise disturb the fit.
- rng_mod=[] : if not empty, a matrix of size [K,I] - a multiplication factor to adjust the range of the quantity K at station I. The actual bounds are just rng*rng_mod
- choose_side_individually : whether to select the left or right tail of the distribution individually for each radial station. If not, we keep the entire left or right side, 
        based on the integral of the value.

Outputs:
extr, p, side
"""
def extrapolate_extremeLoads_curveFit(rng,mat, distr_list, extr_prob, truncThr=None, keepAtLeast=15, logfit=False, 
                                      killUnder=1e-14, rng_mod=[], choose_side_individually=False):
    nbins = np.shape(mat)[2]
    n1 = np.shape(mat)[0]
    n2 = np.shape(mat)[1]

    thr = 1e-5 #threshold (normalized frequency)

    extr = np.zeros((n1,n2))
    extrLR = np.zeros((n1,n2,2))
    paramsLR = np.empty((n1,n2,2),tuple)

    p = np.nan*np.zeros((n1,n2,3))

    if len(rng_mod)==0:
        rng_mod = np.ones((n2,n1))
        
    side = np.ones((n2,n1)) #assuming all the quantities are extrapolated to the right

    for k in range(n2):
        stp_ = (rng[k][1]-rng[k][0])/(nbins)
        x_ = np.arange(rng[k][0]+stp_/2.,rng[k][1],stp_)

        if 'maxForced'  in distr_list[k]:
            for i in range(n1):
                x = x_ * rng_mod[k,i]
                stp = stp_ * rng_mod[k,i]
                #ABSOLUTE MAX: even if it only occurs once (using double precision threshold)
                imax = np.where(mat[i,k,:] >= 1e-16)
                mymax = x[imax[0][-1]] if len(imax[0])>0 else 0.0
                mymin = x[imax[0][0]] if len(imax[0])>0 else 0.0
                extrLR[i,k,0], extrLR[i,k,1] = mymin, mymax
                
                avg, _ = __compute_avg_std(mat[i,k,:], x)
                p[i,k,0] = avg
                p[i,k,1] = mymax - avg
                p[i,k,2] = mymin - avg

        elif 'twiceMaxForced'  in distr_list[k]:
            for i in range(n1):
                x = x_ * rng_mod[k,i]
                stp = stp_ * rng_mod[k,i]

                imax = np.where(mat[i,k,:] >= thr)
                mymax = 2.*x[imax[0][-1]] if len(imax[0])>0 else 0.0
                mymin = 2.*x[imax[0][0]] if len(imax[0])>0 else 0.0
                extrLR[i,k,0], extrLR[i,k,1] = mymin, mymax

                avg, std = __compute_avg_std(mat[i,k,:], x)
                p[i,k,0] = avg
                p[i,k,1] = std #could do min/max instead, as above
                
        elif 'normForced' in distr_list[k]: #a unique normal distribution for both left and right side
            for i in range(n1):
                x = x_ * rng_mod[k,i]
                stp = stp_ * rng_mod[k,i]

                #Curve fitting is a bit sensitive... we could also simply use the good old way.
                # However, it curvefit does not succeed, maybe it is because the distro does not look like a normal at all... 
                #   and would be a good idea not to force that and use a fallback condition instead.
                avg, std = __compute_avg_std(mat[i,k,:], x)
                params = (avg,std)

                mymax = stats.norm.ppf(extr_prob, loc = params[0], scale = params[1])
                mymin = stats.norm.ppf(1.-extr_prob, loc = params[0], scale = params[1])

                extrLR[i,k,0], extrLR[i,k,1] = mymin, mymax
                p[i,k,0] = params[0] #can I do something better than this?
                p[i,k,1] = params[1] #can I do something better than this?
        
        else:  #separate distributions for left and right
            distr = getattr(stats,distr_list[k])

            # Preparing my callbacks for various kinds of distributions, so we can handle 2-params and 3-params distribution fits
            if distr_list[k] in ['norm', 'gumbel_r', 'gumbel_l',]: #--> 2 parameters distributions 
                #the log of the survival function is used for logfit on the right, while the log of the cfd=1-sf is used on thr left
                def logExc_left(x,b,c):
                    return distr.logcdf(x,loc=b,scale=c)     
                def logExc_right(x,b,c):
                    # return np.log(distr.sf(x,loc=b,scale=c))
                    return distr.logsf(x,loc=b,scale=c)                    

                # The percent point function on the left is 1 minus ppf on the right
                def ppf_left(p,a):
                    return distr.ppf(1.-p,loc=a[0],scale=a[1])
                def ppf_right(p,a):
                    return distr.ppf(p,loc=a[0],scale=a[1])
                    
            elif distr_list[k] in ['chi','chi2','weibull_min','weibull_max']: #--> 3 parameters distributions 
                def logExc_left(x,a,b,c):
                    return distr.logcdf(x,a,loc=b,scale=c)     
                def logExc_right(x,a,b,c):
                    return distr.logsf(x,a,loc=b,scale=c)                    

                # The percent point function on the left is 1 minus ppf on the right
                def ppf_left(p,a):
                    return distr.ppf(1.-p,a[0],loc=a[1],scale=a[2])
                def ppf_right(p,a):
                    return distr.ppf(p,a[0],loc=a[1],scale=a[2])
            
            else:
                raise ValueError("I don''t know this distribution. If it's a 2 or a 3 parameter distribution, it should be easy to add it.")
                # To add a distrubution, add its name to the list above in this if clause.

            def ExcDiscr_left(x,stp):
                return np.cumsum(x)*stp
            def ExcDiscr_right(x,stp):
                return 1. - np.cumsum(x)*stp

            logExc = [logExc_left, logExc_right]
            percentile = [ppf_left, ppf_right]
            ExcDiscr = [ExcDiscr_left, ExcDiscr_right]
                   
            # Actually fitting the stuff
            for i in range(n1):
                x = x_ * rng_mod[k,i]
                stp = stp_ * rng_mod[k,i]

                #compute average and std of entire dataset
                avg, std = __compute_avg_std(mat[i,k,:], x)

                #get the threshold, no matter if truncThr is a list or a value or None
                threshold = __list_to_val(truncThr,k)


                #fit and extrapolate for the left and right side separately
                for si,leftright in enumerate([-1,1]):
                    failed = False
                    spn, p0 = __truncate_distro(x, mat[i,k,:], threshold, avg, std, keepAtLeast, leftright, distr_list[k])

                    try: 
                        if logfit:
                            excData = ExcDiscr[si](mat[i,k,:],stp)
                            # print(excData)
                            excData = excData[spn] #truncate 
                            spn2 = np.where(excData>killUnder)[0] #further remove 0.0 and 1e-16 so that we can take the log safely

                            if len(spn2) <= 1:
                                if len(spn2) == 1:
                                    extrLR[i,k,si] = x[spn[spn2]]
                                    print(f"There was only one bin left for a {distr_list[k]} at {k},{i} on the {leftright} side. Will consider the extreme as that value.")
                                elif len(spn2) == 0:
                                    extrLR[i,k,si] = 0.0
                                    print(f"There was no bin left for a {distr_list[k]} at {k},{i} on the {leftright} side. Will consider the extreme as 0.0.")
                                params = (np.nan,0,1)
                                paramsLR[i,k,si] = params
                                continue

                            logExcData = np.log( excData[spn2] ) 
                            
                            params, covf = curve_fit(logExc[si], x[spn[spn2]], logExcData, p0 = p0)  #best possible starting point
                        else:
                            params, covf = curve_fit(distr.pdf, x[spn], mat[i,k,spn], p0 = p0)  #best possible starting point

                        perr = np.sqrt(np.diag(covf))
                        if any(np.isinf(params)) or any(np.isnan(params)) or np.isinf(covf[0][0]) or any(np.isnan(perr)):
                            failed = True

                        extrLR[i,k,si] = percentile[si](extr_prob, params)

                    except TypeError:   
                        # most likely due to the data restriction from spn[spn2] leads to fewer bins than the number of parameters to fit
                        failed = True
                    except RuntimeError:   
                        # most likely failure of the optimization after too many calls
                        failed = True

                    if failed:
                        print(f"Could not determine params for a {distr_list[k]} at {k},{i} on the {leftright} side. Will just double the max load.")
                        imax = np.where(mat[i,k,:] >= thr)

                        if leftright>0:
                            extrLR[i,k,si] = 2.*x[imax[0][-1]] if len(imax[0])>0 else 0.0
                        else:
                            extrLR[i,k,si] = 2.*x[imax[0][0]] if len(imax[0])>0 else 0.0                
                        params = (np.nan,0,1)

                    paramsLR[i,k,si] = params
                
        if choose_side_individually:
            for i in range(n1):
                extr[i,k], side[k,i] = __choose_which_side(extrLR[i,k,0],extrLR[i,k,1])
                si = 0 if side[k,i]<0 else 1

                l = len(paramsLR[i,k,si])
                p[i,k,0:l] = paramsLR[i,k,si][0:l] 

        else:
            integ_left = np.sum(extrLR[:,k,0])
            integ_right = np.sum(extrLR[:,k,1])
            _, leftright = __choose_which_side(integ_left,integ_right)
            si = int(leftright/2 + .5) #convert -1,1 to 0,1
            
            side[k,:] = leftright
            extr[:,k] = extrLR[:,k,si]
            for i in range(n1):
                l = len(paramsLR[i,k,si])
                p[i,k,0:l] = paramsLR[i,k,si][0:l] 
            

    return extr, p, side, extrLR


