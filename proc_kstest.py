# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 16:45:01 2025

@author: youngskim
"""

import numpy as np
import pandas as pd

import temStaPy.distNTS as nts

import pickle
from scipy.stats import norm, kstwobign
from statsmodels.distributions.empirical_distribution import ECDF

# To load the data later:
with open('data/saveddata_NTS_paramest.pickle', 'rb') as f:
    loaded_data = pickle.load(f)
#print(loaded_data)
df_name_weight = loaded_data["df_name_weight"]
sectorsymbols = df_name_weight['symbol'].to_numpy().tolist()
bloombergtickers = df_name_weight['ticker'].to_numpy().tolist()
namerray = df_name_weight['name'].to_numpy().tolist()
excess_sector_returns = loaded_data["sectorescessret"]
strPMNTS = loaded_data["strPMNTS"]
dim = strPMNTS["ndim"]

eta = 0.05

#prior mean
muvec = excess_sector_returns.mean().to_numpy()
#prior Covariance Matrix
cov = excess_sector_returns.cov()

ksg = np.zeros(11)
ksn = np.zeros(11)
pvg = np.zeros(11)
pvn = np.zeros(11)

for n in range(dim):
    data = excess_sector_returns[sectorsymbols[n]]
    zscore = (data-data.mean())/data.std()
    ecdf = ECDF(zscore)
    sorteddata = zscore.sort_values()
    y = ecdf(sorteddata)
    cdf_normal = norm.cdf(sorteddata)
    cdf_stdnts = nts.pnts(sorteddata, ntsparam = np.array([strPMNTS["alpha"], strPMNTS["theta"], strPMNTS["beta"][n]]))
    ksstat_g = max(np.abs(y-cdf_normal))
    ksstat_nts = max(np.abs(y-cdf_stdnts))
    pvalue_g = 1-kstwobign.cdf(ksstat_g*np.sqrt(len(zscore)))
    pvalue_nts = 1-kstwobign.cdf(ksstat_nts*np.sqrt(len(zscore)))
    ksg[n] = ksstat_g
    pvg[n] = pvalue_g
    ksn[n] = ksstat_nts
    pvn[n] = pvalue_nts


dfkstest = pd.DataFrame(columns=['name','ticker',
                                 'K-S(Gaussian)', 'p-value(Gaussian)',
                                 'K-S(NTS)', 'p-value(NTS)' ])

dfkstest["name"]=df_name_weight["name"]
dfkstest["ticker"]=df_name_weight["ticker"]
dfkstest["K-S(Gaussian)"] = ksg
dfkstest["p-value(Gaussian)"] = pvg
dfkstest["K-S(NTS)"] = ksn
dfkstest["p-value(NTS)"] = pvn

print(dfkstest.to_latex(float_format='$%.3f$', index = False)) 

