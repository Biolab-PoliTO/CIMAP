from .targetgraph import targetgraph
from .intervals import intervals

import warnings
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

__all__ = ['modality_distribution']

def modality_distribution(s,target = 'All'):
    
    """
    Method for the graphical representation of the distribution of the cycles into the modalities.

    :Input: * **s** (*dict*): the dictionary containing the cycles activation data created by the Input function from the *.csv* file.
            * **target**: "All" (*default*), labels or portion of the label of the muscle that you want to plot. For example if LGS_R and LGS_L are available inserting the target "LGS" would plot both. Inserting the exact label give as output the graphs related to that label.

    """
    # check that the field "Cycles" exists and is corectly labeled 
    keys = list(s.keys())
    keys_l = [k.lower() for k in keys]
    if not("cycles" in keys_l):
        raise ValueError('"Cycles" key of dictionary not found, check the input dictionary')
    elif not("Cycles" in keys) and "cycles" in keys_l:
        s["Cycles"] = s.pop("cycles")
        warnings.warn('"Cycles" field format wrong, corrected')
        
    # check that the field "Labels" exists and is corectly labeled 
    if not("labels" in keys_l):
        raise ValueError('"Labels" key of dictionary not found, check the input dictionary')
    elif not("Labels" in keys) and "labels" in keys_l:
        s["Labels"] = s.pop("labels")
        warnings.warn(' "Labels" field format wrong, corrected')
    
    toplot = targetgraph(s,target)    
    for i,cycles in enumerate(toplot["Cycles"]):
        plt.figure()
        _,nact,_ = intervals(cycles)
        plt.hist(nact, bins = np.arange(min(nact)-.5,max(nact)+1.5,1), rwidth = .5)
        plt.title(toplot["Labels"][i])
        plt.xlim((0,max(nact)+1))
        plt.xticks(ticks = np.arange(max(nact)+1))
        plt.ylabel('Number of occurrences (#)')
        plt.xlabel('Modalities')