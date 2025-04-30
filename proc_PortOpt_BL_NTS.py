# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 17:43:00 2025

@author: youngskim
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import pickle
from func_util_BLNTS import port_stdev, port_CVaR_Gauss, port_CVaR_NTS, figureout_eff

    
# To load the data later:
with open('data/saveddata_NTS_paramest.pickle', 'rb') as f:
    loaded_data = pickle.load(f)
#print(loaded_data)
df_name_weight = loaded_data["df_name_weight"]
sectorsymbols = df_name_weight['symbol'].to_numpy().tolist()
bloombergtickers = df_name_weight['ticker'].to_numpy().tolist()

with open('data/saveddata_Posterior_BL_NTS.pickle', 'rb') as f:
    loaded_data = pickle.load(f)
#print(loaded_data)
mu_BL = loaded_data["mu_BL"]
mu_BL_CVaR = loaded_data["mu_BL_CVaR"]
sigma_BL= loaded_data["sigma_BL"]
stmv_BL = loaded_data["stmv_BL"]
streg_BL = loaded_data["streg_BL"]
PI_markowitz = loaded_data["PI_markowitz"]
PI_cvar_gauss = loaded_data["PI_cvar_gauss"]
PI_nts = loaded_data["PI_nts"]    
eta = loaded_data["eta"]  

dim = len(mu_BL)
nrep = 100
eqw = np.ones(dim)/dim

### BL Markowitz Optimization
muarray = mu_BL
xG = np.matrix( np.zeros(dim*nrep).reshape(dim, nrep) )
muarG = np.zeros(nrep)
riskarG = np.zeros(nrep)
         

b = 0
cons = ({'type': 'eq', 'fun': lambda x: x.sum() - 1},
        {'type': 'ineq', 'fun': lambda x: x},
        {'type': 'ineq', 'fun': lambda w: np.dot(w, muarray) - b})
res = minimize(
        lambda w: port_stdev(w, sigma_BL),
            x0 = eqw,
            bounds=[(0, 1) for _ in range(dim)],
            constraints = cons,
            options={"maxiter": 1000}
            )
x = res.x.reshape(dim,1)
mustart = (x.T @ muarray.reshape(dim,1)).item()
mu0 = np.linspace(mustart, max(muarray), nrep+1) 
#mu0 = np.linspace(min(muarray), max(muarray), nrep) 
mu0 = mu0[1:]
#mu0 = mu0[1:]

for i in range(mu0.size):
    b = max(mu0[i],0)
    cons = ({'type': 'eq', 'fun': lambda x: x.sum() - 1},
            {'type': 'ineq', 'fun': lambda x: x},
            {'type': 'ineq', 'fun': lambda w: np.dot(w, muarray) - b})
    res = minimize(
        lambda w: port_stdev(w, sigma_BL),
        x0 = eqw,
        bounds=[(0, 1) for _ in range(dim)],
        constraints = cons,
        options={"maxiter": 1000}
    )
    xG[:,i] = res.x.reshape(dim,1)
    muarG[i] = (xG[:,i].T @ muarray.reshape(dim,1)).item()
    riskarG[i] = res.fun

figureout_eff(muarG, riskarG, xG, bloombergtickers, "M-V Optimization")


### BL CVaR Optimization
muarray = mu_BL_CVaR
#mu0 = np.linspace(min(muarray), max(muarray), nrep) 
xC = np.matrix( np.zeros(dim*nrep).reshape(dim, nrep) )
muarC = np.zeros(nrep)
riskarC = np.zeros(nrep)

b = 0
cons = ({'type': 'eq', 'fun': lambda x: x.sum() - 1},
        {'type': 'ineq', 'fun': lambda x: x},
        {'type': 'ineq', 'fun': lambda w: np.dot(w, muarray) - b})
res = minimize(
    lambda w:  port_CVaR_Gauss(w, muarray, sigma_BL, eta),
            x0 = eqw,
            bounds=[(0, 1) for _ in range(dim)],
            constraints = cons,
            options={"maxiter": 1000}
            )
x = res.x.reshape(dim,1)
mustart = (x.T @ muarray.reshape(dim,1)).item()
mu0 = np.linspace(mustart, max(muarray), nrep+1) 
#mu0 = np.linspace(min(muarray), max(muarray), nrep) 
mu0 = mu0[1:]
#mu0 = mu0[1:]

for i in range(mu0.size):
    b = max(mu0[i],0)
    cons = ({'type': 'eq', 'fun': lambda x: x.sum() - 1},
            {'type': 'ineq', 'fun': lambda x: x},
            {'type': 'ineq', 'fun': lambda w: np.dot(w, muarray) - b})
    res = minimize(
        lambda w: port_CVaR_Gauss(w, muarray, sigma_BL, eta),
                x0 = eqw,
                bounds=[(0, 1) for _ in range(dim)],
                constraints=cons,
                options={"maxiter": 1000},
                )
    xC[:,i] = res.x.reshape(dim,1)
    muarC[i] = (xC[:,i].T @ muarray.reshape(dim,1)).item()
    riskarC[i] = res.fun

figureout_eff(muarC, riskarC, xC, bloombergtickers, "Mean-CVaR Optimization")


### BL NTS Optimization
muarray = streg_BL["mu"]
#mu0 = np.linspace(min(muarray), max(muarray), nrep) 
xN = np.matrix( np.zeros(dim*nrep).reshape(dim, nrep) )
muarN = np.zeros(nrep)
riskarN = np.zeros(nrep)

b = 0 #min(muarray)
cons = ({'type': 'eq', 'fun': lambda x: x.sum() - 1},
        {'type': 'ineq', 'fun': lambda x: x},
        {'type': 'ineq', 'fun': lambda w: np.dot(w, muarray) - b})
res = minimize(
    lambda w: port_CVaR_NTS(w, stmv_BL, eta),
    x0 = eqw,
    bounds=[(0, 1) for _ in range(dim)],
    constraints = cons,
    options={"maxiter": 1000}
)
x = res.x.reshape(dim,1)
mustart = (x.T @ muarray.reshape(dim,1)).item()
mu0 = np.linspace(mustart, max(muarray), nrep+1) 
#mu0 = np.linspace(min(muarray), max(muarray), nrep) 
mu0 = mu0[1:]
#mu0 = mu0[1:]

for i in range(mu0.size):
    b = max(mu0[i],0)
    cons = ({'type': 'eq', 'fun': lambda x: x.sum() - 1},
            {'type': 'ineq', 'fun': lambda x: x},
            {'type': 'ineq', 'fun': lambda w: np.dot(w, muarray) - b})
    res = minimize(
        lambda w: port_CVaR_NTS(w, stmv_BL, eta),
        x0 = res.x,
        bounds=[(0, 1) for _ in range(dim)],
        constraints=cons,
        options={"maxiter": 1000},
    )
    xN[:,i] = res.x.reshape(dim,1)
    muarN[i] = (xN[:,i].T @ muarray.reshape(dim,1)).item()
    riskarN[i] = res.fun

figureout_eff(muarN, riskarN, xN, bloombergtickers, "Mean-CVaR-NTS Optimization")


data = {"xG":xG, 
        "xC":xC,
        "xN":xN, 
        "muarG":muarG, 
        "muarC":muarC, 
        "muarN":muarN, 
        "riskarG":riskarG,
        "riskarC":riskarC,
        "riskarN":riskarN,
        "bloombergtickers":bloombergtickers}    
with open('data/saveddata_Portopt_BL_NTS.pickle', 'wb') as f:
    pickle.dump(data, f)