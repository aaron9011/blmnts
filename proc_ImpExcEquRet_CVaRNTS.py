# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 16:18:40 2025

@author: youngskim
"""

import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from numpy.linalg import inv
import func_util_BLNTS as futil
import temStaPy.distMNTS as mnts
import temStaPy.mctrMNTS as fmctrmnts

import pickle
import copy

#def change_mvform2regurla(stmnts):

namerray = ["Communication Services","Consumer Discretionary",
    "Consumer Staples","Energy","Financials",
    "Health Care","Industrials","Information Technology",
    "Materials","Real Estate","Utilities"]

# To load the data later:
with open('data/saveddata_NTS_paramest.pickle', 'rb') as f:
    loaded_data = pickle.load(f)
#print(loaded_data)
df_name_weight = loaded_data["df_name_weight"]
sectorsymbols = df_name_weight['symbol'].to_numpy().tolist()
bloombergtickers = df_name_weight['ticker'].to_numpy().tolist()
namerray = df_name_weight['name'].to_numpy().tolist()
#sectorsymbols = loaded_data["symbol"]
#bloombergtickers = loaded_data["ticker"]
excess_sector_returns = loaded_data["sectorescessret"]
wmkt = loaded_data["w_mkt"]
strPMNTS = loaded_data["strPMNTS"]
dim = strPMNTS["ndim"]

eta = 0.05

#prior mean
#E_wmkt_X = excess_sector_returns.mean().multiply(wmkt['weight'].values).sum()
muvec = excess_sector_returns.mean().to_numpy()
#prior Covariance Matrix
cov = excess_sector_returns.cov()
market_var = np.matmul(wmkt.values.reshape(len(wmkt)).T,
                                       np.matmul(cov.values, wmkt.values.reshape(len(wmkt))))

E_wmkt_X = np.sum(muvec*(wmkt['weight'].values))
#Implied Exceess Equilibrium Markowitz
#risk_aversion = E_wmkt_X / np.sqrt(market_var)
risk_aversion = E_wmkt_X / (market_var)
PI_markowitz = risk_aversion*np.matmul(cov.values, wmkt.values.reshape(len(wmkt)))

#Implied Exceess Equilibrium Normal CVaR
#muvec = excess_sector_returns.mean().to_numpy()
risk_aversion_cvar_gaussian = E_wmkt_X / futil.port_CVaR_Gauss(wmkt.to_numpy().squeeze(), muvec, cov.values, eta)
dSigma = np.array([fmctrmnts.mctStdDev(i, wmkt.to_numpy().squeeze(), cov.values) for i in range(dim)])
dCVaRGauss = futil.CVaR_normal(eta, 0, 1)*dSigma-muvec
dCVaRGauss0 = futil.CVaR_normal(eta, 0, 1)*dSigma
PI_cvar_gauss = dCVaRGauss0*risk_aversion_cvar_gaussian/(2+risk_aversion_cvar_gaussian)

#Implied Exceess Equilibrium NTS CVaR
cvar = fmctrmnts.portfolio_VaR_CVaR_MNTS(eta, wmkt.to_numpy().squeeze(), strPMNTS)
risk_aversion_cvarnts = E_wmkt_X / cvar["CVaRNTS"]
strPMNTS0 =copy.deepcopy(strPMNTS)
strPMNTS0["mu"] = 0*strPMNTS["mu"]
dCVaRNTS = np.array([fmctrmnts.mctCVaR_MNTS(i, eta, wmkt.to_numpy().squeeze(), strPMNTS0) for i in range(dim)])
PI_nts = dCVaRNTS*risk_aversion_cvarnts/(2+risk_aversion_cvarnts)

pringdf = pd.DataFrame(data = {#"Sector":namerray, 
                               "Ticker":bloombergtickers, 
                               "$w_{mkt}$":wmkt['weight']*100, 
                               "$\\Pi_{MV}$":PI_markowitz*10000, 
                               "$\\Pi_{Gauss}$":PI_cvar_gauss*10000, 
                               "$\\Pi_{NTS}$":PI_nts*10000})
pringdf['$w_{mkt}$'] = pringdf['$w_{mkt}$'].map('${:,.2f}$'.format)
pringdf['$\\Pi_{MV}$'] = pringdf['$\\Pi_{MV}$'].map('${:,.4f}$'.format)
pringdf['$\\Pi_{Gauss}$'] = pringdf['$\\Pi_{Gauss}$'].map('${:,.4f}$'.format)
pringdf['$\\Pi_{NTS}$'] = pringdf['$\\Pi_{NTS}$'].map('${:,.4f}$'.format)
print(pringdf.to_latex(index=False))

## Gaussian Prior Distribution
#pri_mean_df = pd.DataFrame( data = {"Ticker":bloombergtickers,
#                                   "$\\mu=\\Pi_{BL}$": PI_markowitz*10000,
#                                   "\\mu = \\PI_{Gauss}": PI_cvar_gauss*10000})
pri_mean_df = pd.DataFrame( data = {"Ticker":bloombergtickers,
                                   "$\\mu$": muvec*10000})
pri_mean_df.set_index("Ticker", inplace = True)
#pri_mean_df['$\\mu=\\Pi_{BL}$'] = pri_mean_df['$\\mu=\\Pi_{BL}$'].map('${:,.4f}$'.format)
print(pri_mean_df.transpose().to_latex(float_format='%.4f'))
cov10000 = 10000*excess_sector_returns.cov()
cov10000.columns = bloombergtickers
cov10000.index = bloombergtickers
print(cov10000.to_latex(float_format='$%.2f$'))


## NTS Prior Distribution
streg = mnts.change_mvform2regform(strPMNTS)
pri_fit_df = pd.DataFrame( data = {"Ticker":bloombergtickers,
                                   "$\\mu$": streg["mu"]*10000,
                          "$\\beta$": streg["beta"]*10000,
                          "$\\gamma$": streg["gamma"]*10000})
pri_fit_df.set_index("Ticker", inplace = True)
pri_fit_df['$\\mu$'] = pri_fit_df['$\\mu$'].map('${:,.4f}$'.format)
pri_fit_df['$\\beta$'] = pri_fit_df['$\\beta$'].map('${:,.4f}$'.format)
pri_fit_df['$\\gamma$'] = pri_fit_df['$\\gamma$'].map('${:,.2f}$'.format)
print(pri_fit_df.transpose().to_latex())

rhodf = pd.DataFrame(data = streg["Rho"])
rhodf.index = bloombergtickers
rhodf.columns = bloombergtickers
print(rhodf.to_latex(float_format='$%.2f$'))
print(np.around(streg["alpha"],4))
print(np.around(streg["theta"],4))



## Black Litterman Start
v = np.array([9.5,1.5,1.5,2])/100/250 #change annual return to daily return
P = np.asarray([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, -1, 0]])
tau = 1
omega = futil.error_cov_matrix(cov, tau, P)
dfomega10000 = pd.DataFrame(data = omega*10000)
print(dfomega10000.to_latex(float_format='%.4f', index=False))

### BL Original Posterior
sigmtx = cov.to_numpy()
mu_BL = PI_markowitz + sigmtx@ P.T @ inv(P @ sigmtx @ P.T+omega) @ (v - P @ PI_markowitz)
mu_BL_CVaR = PI_cvar_gauss + sigmtx@ P.T @ inv(P @ sigmtx @ P.T+omega) @ (v - P @ PI_cvar_gauss)
sigma_BL = sigmtx - sigmtx@ P.T @ inv(P @ sigmtx @ P.T+omega) @ P @ sigmtx

post_mean_df = pd.DataFrame( data = {"Ticker":bloombergtickers,
                                   "$\\mu_{MV}$": mu_BL*10000,
                                   "$\\mu_{Gauss}$": mu_BL_CVaR*10000})
post_mean_df.set_index("Ticker", inplace = True)
#post_mean_df['$\\mu_{BL}$'] = post_mean_df['$\\mu_{BL}$'].map('${:,.4f}$'.format)
print(post_mean_df.transpose().to_latex(float_format='$%.4f$'))
cov10000 = pd.DataFrame( data = 10000*sigma_BL )
cov10000.columns = bloombergtickers
cov10000.index = bloombergtickers
print(cov10000.to_latex(float_format='$%.2f$'))

#### BL MNTS 
strPMNTS["mu"] = PI_nts.reshape(len(PI_nts),1).squeeze()
stmv_BL, streg_BL = futil.get_stmnts4BL(strPMNTS, P, omega, v)

## print posterior NTS
prost_fit_df = pd.DataFrame( data = {"Ticker":bloombergtickers,
                                   "$\\mu^*$": streg_BL["mu"]*10000,
                          "$\\beta^*$": streg_BL["beta"]*10000,
                          "$\\gamma^*$": streg_BL["gamma"]*10000})
prost_fit_df.set_index("Ticker", inplace = True)
prost_fit_df['$\\mu^*$'] = prost_fit_df['$\\mu^*$'].map('${:,.4f}$'.format)
prost_fit_df['$\\beta^*$'] = prost_fit_df['$\\beta^*$'].map('${:,.4f}$'.format)
prost_fit_df['$\\gamma^*$'] = prost_fit_df['$\\gamma^*$'].map('${:,.2f}$'.format)
print(prost_fit_df.transpose().to_latex())

rhodf = pd.DataFrame(data = streg_BL["Rho"])
rhodf.index = bloombergtickers
rhodf.columns = bloombergtickers
print(rhodf.to_latex(float_format='$%.2f$'))

print(np.around(streg_BL["alpha"],4))
print(np.around(streg_BL["theta"],4))


data = {"mu_BL":mu_BL, 
        "mu_BL_CVaR":mu_BL_CVaR,
        "sigma_BL":sigma_BL, 
        "stmv_BL":stmv_BL, 
        "streg_BL":streg_BL,
        "PI_markowitz":PI_markowitz,
        "PI_cvar_gauss":PI_cvar_gauss,
        "PI_nts":PI_nts,
        "eta":eta}    
with open('data/saveddata_Posterior_BL_NTS.pickle', 'wb') as f:
    pickle.dump(data, f)




