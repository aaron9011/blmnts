# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 12:45:27 2025

@author: youngskim
"""


import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

import pickle
from PIL import Image


isPosterior = True
#isPosterior = False


                    
if (isPosterior):
    with open('data/saveddata_Portopt_BL_NTS.pickle', 'rb') as f:
        loaded_data = pickle.load(f)
    legfla = True
    savefigfilename = "fig/opt_Posterior"
    titlestrlst = ["b1\nMean-Variance Optimization", "b2\nMean-CVaR Optimization", "b3\nMean-CVaR-NTS Optimization"]
else:
    with open('data/saveddata_Portopt_Prior_BL_NTS.pickle', 'rb') as f:
        loaded_data = pickle.load(f)
    legfla = False
    savefigfilename = "fig/opt_Prior"
    titlestrlst = ["a1\nMean-Variance Optimization", "a2\nMean-CVaR Optimization", "a3\nMean-CVaR-NTS Optimization"]


xG = loaded_data['xG'] 
xC = loaded_data['xC'] 
xN = loaded_data['xN'] 
muarG = loaded_data['muarG'] 
muarC = loaded_data['muarC'] 
muarN = loaded_data['muarN'] 
riskarG = loaded_data['riskarG'] 
riskarC = loaded_data['riskarC'] 
riskarN = loaded_data['riskarN'] 
bloombergtickers = loaded_data['bloombergtickers'] 


idx = (muarG>(min(muarG))) & (muarG<max(muarG))
xG = xG[:,idx]
muarG = muarG[idx]
idx = (muarC>(min(muarC))) & (muarC<max(muarC))
xC = xC[:,idx]
muarC = muarC[idx]
idx = (muarN>(min(muarN))) & (muarN<max(muarN))
xN = xN[:,idx]
muarN = muarN[idx]

xlst = [xG, xC, xN]
mulst = [muarG, muarC, muarN]
dflst = []
titlestr = "lala"

fig, axes = plt.subplots(1, 3, figsize=(10, 10), sharex=True)

title_font = {'fontsize': 26, 'fontweight': 'bold', 'fontname': 'Arial'}
for n in range(3):
    df = pd.DataFrame(data = xlst[n], index = bloombergtickers)
    df.columns = mulst[n]*10000
    df.columns = df.columns.map('${:,.4f}$'.format)
    dftr = df.transpose()
    dflst.append(dftr)
    dflst[n].plot(ax=axes[n],  kind='bar', stacked=True, width=1, 
                  colormap = "Pastel1",
             figsize=(20, 10), legend = False)
    axes[n].set_title(titlestrlst[n], fontdict=title_font)
    axes[n].set_xlabel('Expected Excess Return')
    axes[n].set_ylabel('Proportion')
    
    hatches = [ '//', 'oo', '\\\\', '..', '||', '','--', 'xx', 'OO', '++','**' ]
    
    # Iterating through bars and assigning hatches
    for i, container in enumerate(axes[n].containers):
        hatch = hatches[i % len(hatches)]
        for patch in container.patches:
            patch.set_hatch(hatch)
    selected_indices = list(range(0, len(df.columns), 10))  # Show every 2nd label
    axes[n].set_xticks(selected_indices)
    axes[n].set_xticklabels(df.columns[selected_indices]) 

if (legfla) :
     fig.legend( bloombergtickers,loc='lower center',  bbox_to_anchor=(0.5, -0.10), 
           ncol = 6, borderaxespad=0, fontsize="20")

plt.tight_layout()

plt.savefig(savefigfilename+".eps", format="eps", bbox_inches='tight') # Save as EPS
plt.savefig(savefigfilename+".pdf", format="pdf", bbox_inches='tight') # Save as PDF
plt.savefig(savefigfilename+".png", format="png", bbox_inches='tight') # Save as PDF
plt.show()
image = Image.open(savefigfilename+".png").convert('L')
image.save(savefigfilename+"gray.png")

