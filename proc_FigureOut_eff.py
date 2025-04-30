# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 12:24:16 2025

@author: youngskim
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

import pickle
from PIL import Image
    
    
def figureout_eff(muar, riskar, titlestr, xaxisstr, savefilename):
    idx = (muar>(min(muar))) & (muar<max(muar))
    plt.plot(riskar[idx]*100, muarG[idx]*10000) 
    plt.title(titlestr)
    plt.xlabel(xaxisstr)
    plt.ylabel('Expected Excess Return')
    plt.savefig(savefilename+".eps", format="eps") # Save as EPS
    plt.savefig(savefilename+".pdf", format="pdf") # Save as PDF
    plt.savefig(savefilename+".png", format="png") # Save as PDF
    plt.show()
    image = Image.open(savefilename+".png").convert('L')
    image.save(savefilename+"gray.png")

    

def figureout_weightvec(muar,  xmtx, bloombergtickers, titlestr, savefilename, legfla = True):
    idx = (muar>(min(muar))) & (muar<max(muar))
    df = pd.DataFrame(data = xmtx[:,idx])
    df.index = bloombergtickers
    df.columns = muar[idx]*10000
    df.columns = df.columns.map('${:,.4f}$'.format)

    df_sorted = df.transpose()
    ax = df_sorted.plot( kind='bar', stacked=True, width=1, colormap = "Pastel1",
             figsize=(6, 10), title=titlestr, legend = legfla)
    ax.set_xlabel('Expected Excess Return')
    ax.set_ylabel('Proportion')
    
    hatches = [ '//', 'oo', '\\\\', '..', '||', '','--', 'xx', 'OO', '++','**' ]
    
    # Iterating through bars and assigning hatches
    for i, container in enumerate(ax.containers):
        hatch = hatches[i % len(hatches)]
        for patch in container.patches:
            patch.set_hatch(hatch)
    if (legfla == True) :
        ax.legend(bbox_to_anchor=(1.04, 0.5), borderaxespad=0, fontsize="20")
    
    selected_indices = list(range(0, len(df.columns), 10))  # Show every 2nd label
    ax.set_xticks(selected_indices)
    ax.set_xticklabels(df.columns[selected_indices]) 
    plt.savefig(savefilename+".eps", format="eps", bbox_inches='tight') # Save as EPS
    plt.savefig(savefilename+".pdf", format="pdf", bbox_inches='tight') # Save as PDF
    plt.savefig(savefilename+".png", format="png", bbox_inches='tight') # Save as PDF
    plt.show()
    image = Image.open(savefilename+".png").convert('L')
    image.save(savefilename+"gray.png")
    return ax
    
    
with open('data/saveddata_Portopt_BL_NTS.pickle', 'rb') as f:
    loaded_data = pickle.load(f)
    
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

figureout_eff(muarG, riskarG,
              "Mean-Variance Optimization",
              'Postfolio Standard Deviation',
              'fig/effMV')

figureout_eff(muarC, riskarC,
              "Mean-CVaR Optimization",
              'CVaR',
              'fig/effMCVaR')

figureout_eff(muarN, riskarN,
              "Mean-CVaR-NTS Optimization",
              'CVaR',
              'fig/effMCVaRNTS')


figureout_weightvec(muarG, xG, bloombergtickers, 
                    "Mean-Variance Optimization", 
                    'fig/AllWeightMV', False)
figureout_weightvec(muarC, xC, bloombergtickers, 
                    "Mean-CVaR Optimization", 
                    'fig/AllWeightCVaR', False)
figureout_weightvec(muarN, xN, bloombergtickers, 
                    "Mean-CVaR-NTS Optimization", 
                    'fig/AllWeightCVaRNTS', True)

with open('data/saveddata_Portopt_Prior_BL_NTS.pickle', 'rb') as f:
    loaded_data = pickle.load(f)
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
    
figureout_weightvec(muarG, xG, bloombergtickers, 
                    "Mean-Variance Optimization", 
                    'fig/AllWeightMV_prior', False)
figureout_weightvec(muarC, xC, bloombergtickers, 
                    "Mean-CVaR Optimization", 
                    'fig/AllWeightCVaR_prior', False)
figureout_weightvec(muarN, xN, bloombergtickers, 
                    "Mean-CVaR-NTS Optimization", 
                    'fig/AllWeightCVaRNTS_prior', False)


