# -*- coding: utf-8 -*-
"""
Created on Fri Feb 03 14:25:59 2017

@author: wexiao
"""

import time

import numpy as np
import torch 
torch.cuda.empty_cache()
from models.rpca.utility import thres
import gc
gc.collect()

def spca(M, lam=np.nan, mu=np.nan, tol=10**(-7), maxit=1000, verbose=True):
    """ Stable Principal Component Pursuit (Zhou et al., 2009)
    
    This code solves the following optimization problem
    min_{L,S} { ||L||_* + lam*||S(:)||_1 + 1/{2*mu}||M-L-S||_F^2}
    using the Accelerated Proximal Gradient method with a fixed mu_iter.
  
    Parameters
    ----------
    M : array-like, shape (n_features, n_samples), which will be decomposed into a sparse matrix S 
        and a low-rank matrix L.
        
    lam : positive tuning parameter (default NaN). When lam is set to NaN,  the value 1/sqrt(max(m, n)) * factor 
    will be used in the algorithm, where M is a m by n matrix.
    
    mu: postive tuning parameter (default NaN). When mu is set to NaN, the value sqrt(2*max(m, n)) 
    will be used in the algorithm, where M is a m by n matrix. A good choice of mu is sqrt(2*max(m, n))*sigma,
    where sigma is the standard deviation of error term.
    
    tol : tolerance value for convergency (default 10^-7).
    
    maxit : maximum iteration (default 1000).

    verbose : bool
    
    Returns
    ----------
    L1 : array-like, low-rank matrix.
    
    S1 : array-like, sparse matrix.
    
    k : number of iteration.
    
    rank : rank of low-rank matrix.
    
    References
    ----------
    Zhou, Zihan, et al. "Stable principal component pursuit." 
        Information Theory Proceedings (ISIT), 2010 IEEE International Symposium on. IEEE, 2010.
    
    Lin, Zhouchen, et al. "Fast convex optimization algorithms for exact recovery of a corrupted low-rank matrix." 
    Computational Advances in Multi-Sensor Adaptive Processing (CAMSAP) 61.6 (2009).
    
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    M = M.to(device)
    # parameter setting
    m, n = M.shape
    if np.isnan(mu):
        mu = np.sqrt(2*max(m,n))
    if np.isnan(lam):
        lam = 1.0/np.sqrt(max(m,n))
    
    # initialization
    L0 = torch.zeros((m,n)) 
    L0 = L0.to(device)
    L1 = torch.zeros((m,n)) 
    L1 = L0.to(device)
    S0 = torch.zeros((m,n))
    S0 = S0.to(device)
    S1 = torch.zeros((m,n))
    S1 = S1.to(device)
    t0 = 1
    t1 = 1
    mu_iter = mu
    k = 1
    time_flag = None
    
    counter = 0
    while 1:
        if verbose:
            time_flag = time.time()
        Y_L = L1 + (t0-1)/t1*(L1-L0)
        Y_L = Y_L.to(device)
        Y_S = S1 + (t0-1)/t1*(S1-S0)
        Y_S = Y_S.to(device)
        G_L = Y_L - 0.5*(Y_L + Y_S - M)
        G_L = G_L.to(device)
        U, sigmas, V = torch.linalg.svd(G_L, full_matrices=False);
        U = U.to(device)
        sigmas = sigmas.to(device)
        V = V.to(device)
        rank = (sigmas > mu_iter/2).sum()
        Sigma = torch.diag(sigmas[0:rank] - mu_iter/2)
        Sigma.to(device)
        L0 = L1
        L0 = L0.to(device)
        L1 = torch.mm(torch.mm(U[:,0:rank], Sigma), V[0:rank,:])
        L1 = L1.to(device)
        G_S = Y_S - 0.5*(Y_L + Y_S - M)
        G_S = G_S.to(device)
        S0 = S1
        S0 = S0.to(device)
        lamu = lam*mu_iter/2*torch.ones(G_S.shape)
        lamu = lamu.to(device)
        S1 = thres(G_S, lamu)
        t1, t0 = (np.sqrt(t1**2+1) + 1)/2, t1
        
        # stop the algorithm when converge
        E_L =2*(Y_L - L1) + (L1 + S1 - Y_L - Y_S)
        E_L = E_L.to(device)
        E_S =2*(Y_S - S1) + (L1 + S1 - Y_L - Y_S) 
        E_S = E_S.to(device)
        dist = np.sqrt(torch.linalg.norm(E_L, ord='fro').item()**2 + torch.linalg.norm(E_S, ord='fro').item()**2)
        if verbose and k % 5 == 0:
            print(f"iter {k}: took {time.time() - time_flag:.2f} seconds, dist = {dist}", flush=True)
        if k >= maxit or dist < tol:
            print("RPCA done")
            break
        else:
            k += 1
        counter += 1
        if (counter % 10 == 0):
            print(str(counter//10)+" iterations of RPCA")   
            print(rank)
    return L1, S1, k, rank
        

        
    
    
    
    
    
    
    
    
    
    
    
    
