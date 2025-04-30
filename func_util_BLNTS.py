#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 10:39:52 2025

@author: aaronkim
"""


import yfinance as yahooFinance
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from numpy.linalg import inv
from scipy.stats import norm
import temStaPy.distMNTS as mnts
import temStaPy.mctrMNTS as fmctrmnts

def implied_rets(risk_aversion, sigma, w):
    
    implied_rets = risk_aversion * sigma.dot(w).squeeze()
    
    return implied_rets

def error_cov_matrix(sigma, tau, P):
    matrix = np.diag(np.diag(P.dot(tau * sigma).dot(P.T)))
    return matrix

def get_closeprice_fromYF(startDate,endDate,symbols):
    spxdf = yahooFinance.download(symbols, start=startDate, end=endDate)
    spxdf.sort_values(by=['Date'], inplace=True, ascending=True)
    spxdf.reset_index(inplace=True)  # Make it no longer an Index
    spxdf['Date'] = pd.to_datetime(spxdf['Date'], format="%Y/%m/%d") 
    dfprice = spxdf["Close"]
    dfprice = dfprice.set_index(spxdf["Date"])
    
    npret = np.diff(np.log(dfprice.to_numpy().T), axis = 1).T
    datearray = dfprice.index.delete(0)

    dfret = pd.DataFrame(npret)
    dfret = dfret.set_index(datearray)
    dfret.columns = dfprice.columns
    
    return dfprice,dfret

def CVaR_normal(eta, mu, sigma, dt=1):
    #VaR = -norm.ppf(eps, loc=mu * dt, scale=sigma * np.sqrt(dt))
    stdK = norm.ppf(eta)
    avar = sigma * np.sqrt(dt) / (eta * np.sqrt(2 * np.pi)) * np.exp(-stdK**2 / 2) - mu * dt
    return avar.item()

#def CVaR_Port_Normal(w, eta, muvec, SigmaMtx):
#    portvar =  w.T @ SigmaMtx @ w
#    portmean = w.T @ muvec
#    return CVaR_normal(eta, portmean.flatten(), np.sqrt(portvar.flatten()))

def get_stmnts4BL(stmnts, P, omega, v, formflag = 0):
    if(formflag == 0): #inputvariable is mv form
        streg = mnts.change_mvform2regform(stmnts)    
    else:
        streg = stmnts
        
    sigmtx = np.diag(streg["gamma"]) @ streg["Rho"] @ np.diag(streg["gamma"])
    vvec = v.reshape(len(v),1)
    muvec = streg["mu"].reshape(len(streg["mu"]),1)
    betavec = streg["beta"].reshape(len(streg["beta"]),1)

    #mustar = muvec-betavec + sigmtx@ P.T @ inv(P @ sigmtx @ P.T+omega) @ (vvec - P @ (muvec-betavec))
    mustar = muvec + sigmtx@ P.T @ inv(P @ sigmtx @ P.T+omega) @ (vvec - P @ muvec)
    betasatr = betavec - sigmtx@ P.T @ inv(P @ sigmtx @ P.T+omega) @ P @ betavec
    sigStarBL = sigmtx - sigmtx@ P.T @ inv(P @ sigmtx @ P.T+omega) @ P @ sigmtx
    gamstar = np.sqrt(np.diag(sigStarBL))
    rhostar = np.diag(gamstar**(-1)) @ sigStarBL @ np.diag(gamstar**(-1))

    streg_BL = {
        "ndim": streg["ndim"],
        "alpha": streg["alpha"],
        "theta": streg["theta"],
        "mu": mustar.squeeze(),
        "beta": betasatr.squeeze(),
        "gamma": gamstar,
        "Rho": rhostar,
        "CovMtx": streg["CovMtx"]
    }
    stmv_BL = mnts.change_regform2mvform(streg_BL)
    return stmv_BL, streg_BL

def port_stdev(w, Sigma):
    w = w.reshape(len(w),1)
    return np.sqrt(w.T @ Sigma @ w)

def port_CVaR_Gauss(w, mu, Sigma, alpha):
    w = w.reshape(len(w),1)
    mu = mu.reshape(len(w),1)
    m = w.T @ mu
    s = np.sqrt(w.T @ Sigma @ w)
    CVaRGauss = CVaR_normal(alpha, 0, 1)*s-m
    return CVaRGauss.squeeze()

def port_CVaR_NTS(w, stmnts, alpha):
    res = fmctrmnts.portfolio_VaR_CVaR_MNTS(alpha, w, stmnts)
    return res["CVaRNTS"]
    
def figureout_eff(muar, riskar, xmtx, bloombergtickers, titlestr):
    idx = (muar>(min(muar))) & (muar<max(muar))
    plt.plot(riskar[idx]*100, 
             muar[idx]*10000)

    df = pd.DataFrame(data = xmtx[:,idx])
    df.index = bloombergtickers
    df.columns = muar[idx]*10000
    df.columns = df.columns.map('${:,.4f}$'.format)

    df_sorted = df.transpose()
    ax = df_sorted.plot( kind='bar', stacked=True, width=1, colormap = "Pastel2",
             figsize=(6, 10),
             title=titlestr)
    hatches = [ '//', 'oo', '\\\\', '..', '||', ' ','--', 'xx', 'OO', '++','**' ]
    
    # Iterating through bars and assigning hatches
    for i, container in enumerate(ax.containers):
        hatch = hatches[i % len(hatches)]
        for patch in container.patches:
            patch.set_hatch(hatch)
    ax.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0, fontsize="20")
    
    selected_indices = list(range(0, len(df.columns), 10))  # Show every 2nd label
    ax.set_xticks(selected_indices)
    ax.set_xticklabels(df.columns[selected_indices]) 
    plt.show()

#def constraint_eq(x):
#    return x.sum() - 1

# Inequality constraint example: x[0] >= 0
#def constraint_ineq(x):
#    return x

