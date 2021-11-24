import numpy as np
from scipy import stats
from scipy.optimize import curve_fit


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

    return EXTR_life_B1, p
    

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
            
    return extr, p


def extrapolate_extremeLoads_curveFit(rng,mat,distr_list, extr_prob, truncThr=None, keepAtLeast=5, logfit=False, killUnder=1e-14):
    nbins = np.shape(mat)[2]
    n1 = np.shape(mat)[0]
    n2 = np.shape(mat)[1]

    thr = 1e-5 #threshold (normalized frequency)

    extr = np.zeros((n1,n2))

    p = np.nan*np.zeros((n1,n2,3)) #not very general ...

    for k in range(n2):
        stp = (rng[k][1]-rng[k][0])/(nbins)
        x = np.arange(rng[k][0]+stp/2.,rng[k][1],stp)
        
        if 'twiceMaxForced'  in distr_list[k]:
            for i in range(n1):
                imax = np.where(mat[i,k,:] >= thr)
                extr[i,k] = 2.*x[imax[0][-1]] if len(imax[0])>0 else 0.0
                
        elif 'normForced' in distr_list[k]:
            for i in range(n1):
                #Curve fitting is a bit sensitive... we could also simply use the good old way.
                # However, it curvefit does not succeed, maybe it is because the distro does not look like a normal at all... 
                #   and would be a good idea not to force that and use a fallback condition instead.
                avg = np.sum( mat[i,k,:] * x ) / np.sum(mat[i,k,:])
                std = np.sqrt( np.sum( mat[i,k,:] * x**2 ) / np.sum(mat[i,k,:]) - avg )
                params = (avg,std)

                extr[i,k] = stats.norm.ppf(extr_prob, loc = params[0], scale = params[1])
                p[i,k,0] = params[0] #can I do something better than this?
                p[i,k,1] = params[1] #can I do something better than this?
        elif 'norm' in distr_list[k] or 'gumbel' in distr_list[k]: #--> 2 parameters distributions 
            distr = getattr(stats,distr_list[k])
            for i in range(n1):
                failed = False

                #compute average and std of entire dataset
                avg = np.sum( mat[i,k,:] * x ) / np.sum(mat[i,k,:])
                std = np.sqrt( np.sum( mat[i,k,:] * x**2 ) / np.sum(mat[i,k,:]) - avg )
                #compute the range over which the fit must be done, if user asked to reduce the dataset
                if truncThr is None or (isinstance(truncThr,list) and ( (len(truncThr)==0) or (truncThr[k] is None) )):
                    #don't trash or truncate anything
                    spn = np.array(range(len(mat[i,k,:])))
                    p0 = [avg,std] #initial search point
                elif isinstance(truncThr,list):
                    spn = np.where(x>=avg+truncThr[k]*std)[0]
                    p0 = [avg+truncThr[k]*std,std/truncThr[k]] #move initial search point towards the tail
                else:
                    spn = np.where(x>=avg+truncThr*std)[0]
                    p0 = [avg+truncThr*std,std/truncThr] #move initial search point towards the tail
                if keepAtLeast>0:
                    if len(spn) == 0:
                        print("Warning: empty set. Reintroduce the whole dataset.")
                        spn = range(len(mat[i,k,:]))
                    elif len(spn)<keepAtLeast:
                        spn = range(spn[-1]-keepAtLeast,spn[-1]+1)

                try: 
                    if logfit:
                        excData = 1.0 - np.cumsum(mat[i,k,:])*stp
                        excData = excData[spn] #truncate 
                        spn2 = np.where(excData>killUnder)[0] #further remove 0.0 and 1e-16 so that we can take the log safely
                        logExcData = np.log( excData[spn2] ) 
                        def logExc(x,b,c):
                            return np.log(distr.sf(x,loc=b,scale=c))
                        params, covf = curve_fit(logExc, x[spn[spn2]], logExcData, p0 = p0)  #best possible starting point
                    else:
                        params, covf = curve_fit(distr.pdf, x[spn], mat[i,k,spn], p0 = p0)  #best possible starting point
                    perr = np.sqrt(np.diag(covf))
                    if any(np.isinf(params)) or any(np.isnan(params)) or np.isinf(covf[0][0]) or any(np.isnan(perr)):
                        failed = True
                    extr[i,k] = distr.ppf(extr_prob, loc = params[0], scale = params[1])
                except Exception:   
                    failed = True

                if failed:
                    print(f"Could not determine params for a {distr_list[k]} at {k},{i}. Will just double the max load.")
                    imax = np.where(mat[i,k,:] >= thr)
                    extr[i,k] = 2.*x[imax[0][-1]] if len(imax[0])>0 else 0.0
                    params = (np.nan,0,1)
                
                p[i,k,0] = params[0] #can I do something better than this?
                p[i,k,1] = params[1] #can I do something better than this?
        else: #--> 3 parameters distributions: chi2, weibul
            distr = getattr(stats,distr_list[k])
            for i in range(n1):
                failed = False

                #compute average and std of entire dataset
                avg = np.sum( mat[i,k,:] * x ) / np.sum(mat[i,k,:])
                std = np.sqrt( np.sum( mat[i,k,:] * x**2 ) / np.sum(mat[i,k,:]) - avg )
                #compute the range over which the fit must be done, if user asked to reduce the dataset
                if truncThr is None or (isinstance(truncThr,list) and ( (len(truncThr)==0) or (truncThr[k] is None) )):
                    spn = np.array(range(len(mat[i,k,:])))
                    #initial search point -- black magic
                    # p0=[5,0,1000]  
                    # p0=[50,0,50]    #init conditions: works well but not all the time
                    p0=[20,-avg/2,std/2]  #empirical way of determining initial condition, works ok with chi2
                elif isinstance(truncThr, list):
                    spn = np.where(x>=avg+truncThr[k]*std)[0]
                    p0=[5,avg,std/2]  #initial search point -- black magic. Works well for aero loads and moderate truncation.
                else:
                    spn = np.where(x>=avg+truncThr*std)[0]
                    p0=[5,avg,std/2]  #initial search point -- black magic. Works well for aero loads and moderate truncation.

                if keepAtLeast>0:
                    if len(spn) == 0:
                        print("Warning: empty set. Reintroduce the whole dataset.")
                        spn = range(len(mat[i,k,:]))
                    elif len(spn)<keepAtLeast:
                        spn = range(spn[-1]-keepAtLeast,spn[-1]+1)

                try:
                    if logfit:
                        excData = 1.0 - np.cumsum(mat[i,k,:])*stp
                        excData = excData[spn] #truncate 
                        spn2 = np.where(excData>killUnder)[0] #further remove 0.0 and 1e-16 so that we can take the log safely
                        logExcData = np.log( excData[spn2] ) 
                        def logExc(x,a,b,c):
                            return np.log(distr.sf(x,a,loc=b,scale=c))
                        params, covf = curve_fit(logExc, x[spn[spn2]], logExcData, p0 = p0)  #best possible starting point
                    else:
                        params, covf = curve_fit(distr.pdf, x[spn], mat[i,k,spn], p0=p0) 
                    
                    perr = np.sqrt(np.diag(covf))
                    if any(np.isinf(params)) or any(np.isnan(params)) or np.isinf(covf[0][0]) or any(np.isnan(perr)):
                        failed = True
                    #     raise RuntimeError("")
                    # print(f"{k},{i}: {perr}")
                    extr[i,k] = distr.ppf(extr_prob, params[0], loc=params[1], scale=params[2])
                except Exception:
                    failed = True
                    
                if failed:
                    print(f"Could not determine params for a {distr_list[k]} at {k},{i}. Will just double the max load.")
                    imax = np.where(mat[i,k,:] >= thr)
                    extr[i,k] = 2.*x[imax[0][-1]] if len(imax[0])>0 else 0.0
                    params = (np.nan,0,1)

                p[i,k,0] = params[0] #can I do something better than this?
                p[i,k,1] = params[1] #can I do something better than this?
                p[i,k,2] = params[2] #can I do something better than this?

    return extr, p