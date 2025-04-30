# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 16:27:01 2025

@author: youngskim

sector_symbols = {
    "Communication Services": "^SP500-50",
    "Consumer Discretionary": "^SP500-25",
    "Consumer Staples": "^SP500-30",
    "Energy": "^GSPE",
    "Financials": "^SP500-40",
    "Health Care": "^SP500-35",
    "Industrials": "^SP500-20",
    "Information Technology": "^SP500-45",
    "Materials": "^SP500-15",
    "Real Estate": "^SP500-60",
    "Utilities": "^SP500-55",
    "13 WEEK TREASURY BILL": "^IRX"
}

Communication Services	S5TELS
Consumer Discretionary	S5COND
Consumer Staples	S5CONS
Energy	S5ENRS
Financials	S5FINL
Health Care	S5HLTH
Industrials	S5INDU
Information Technology	S5INFT
Materials	S5MATR    
Real Estate	S5RLST
Utilities	S5UTIL

"""

import datetime
import pandas as pd

import pickle

from temStaPy.distNTS import fitstdnts
import temStaPy.distMNTS as mnts

import func_util_BLNTS as futil

sectornames = ["Communication Services",
    "Consumer Discretionary",
    "Consumer Staples",
    "Energy",
    "Financials",
    "Health Care",
    "Industrials",
    "Information Technology",
    "Materials",
    "Real Estate",
    "Utilities"]
sectorsymbols = ["^SP500-50","^SP500-25","^SP500-30","^GSPE","^SP500-40","^SP500-35",
           "^SP500-20","^SP500-45","^SP500-15","^SP500-60","^SP500-55"]
bloombergtickers = ["S5TELS","S5COND","S5CONS","S5ENRS","S5FINL","S5HLTH",
           "S5INDU","S5INFT","S5MATR","S5RLST","S5UTIL"]
sectorweightvector = [9.45, 10.5, 5.88, 3.30, 14.52, 10.77, 8.32, 30.69, 1.99, 2.19, 2.39]
#https://en.macromicro.me/collections/34/us-stock-relative/121244/sp-500-gics-sectors-weightings-monthly

df_name_weight = pd.DataFrame( data = {'name':sectornames, 
                                       'symbol':sectorsymbols, 
                                       'ticker':bloombergtickers, 
                                       'weight':sectorweightvector} )

df_name_weight=df_name_weight.sort_values(by='weight', ascending = False)

sectorsymbols = df_name_weight['symbol'].to_numpy().tolist()
bloombergtickers = df_name_weight['ticker'].to_numpy().tolist()
sectorweightvector = df_name_weight['weight'].to_numpy().tolist()


wmkt = pd.DataFrame({"weight":sectorweightvector}, index = sectorsymbols)/sum(sectorweightvector)
treasurysymbol = ["^IRX"]
dim = len(sectorsymbols)
eta = 0.05

# startDate , as per our convenience we can modify
startDate = datetime.datetime(2015, 3, 1)
# endDate , as per our convenience we can modify
endDate = datetime.datetime(2025, 2, 28)


sectordatadf,dfsecret = futil.get_closeprice_fromYF(startDate,endDate, sectorsymbols)
r_rf = futil.get_closeprice_fromYF(startDate,endDate, treasurysymbol)[0]
#match date of those two data
mg = r_rf.join(dfsecret, how='inner') 
r_rf_daily = mg["^IRX"]/360.0/100 * (250/360)
dfsecret = mg[sectorsymbols]
excess_sector_returns = dfsecret.subtract(r_rf_daily, axis=0).dropna()

pr_sp500, ret_sp500 = futil.get_closeprice_fromYF(startDate,endDate, ["^GSPC"])
excess_indexret = ret_sp500.subtract(r_rf_daily, axis=0).dropna()
rawdat = (excess_indexret-ret_sp500.mean())/ret_sp500.std()
stdntsparam = fitstdnts(rawdat.to_numpy().squeeze())

strPMNTS = mnts.fitmnts(excess_sector_returns.to_numpy(), dim, alphaNtheta=stdntsparam[0:2])

data = {"df_name_weight":df_name_weight,
        "sectorescessret":excess_sector_returns, 
        "w_mkt":wmkt, 
        "strPMNTS":strPMNTS}    
with open('data/saveddata_NTS_paramest.pickle', 'wb') as f:
    pickle.dump(data, f)
    

