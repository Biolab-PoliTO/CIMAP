from .targetgraph import targetgraph
import warnings
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

__all__ = ['actplot']

def actplot(s,target = 'All'):
    """
    Function for plotting all the cycles and the activation before the application of CIMAP
    
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
    for i,cyc in enumerate(toplot["Cycles"]):
        # setting the gaps to NaN so that it won't be graphed in any way and leaves
        # an empty space
        plt.figure(figsize=(6,6))
        cycles = cyc.copy()
        cycles[cycles == 0] = np.nan
        X = np.linspace(0,100,cycles.shape[1])
        for j,cyc in enumerate(cycles):
            plt.plot(X,cyc*-j,'b', linewidth=2)
        plt.ylim((-j-1,1))
        plt.xlim((0,100))
        plt.title(toplot["Labels"][i])
        plt.yticks(ticks = np.arange(-j,1), labels = np.arange(j+1,0,-1),fontsize=6)
        plt.grid(alpha = 0.6)
        plt.ylabel('Gait cycles')
        plt.xlabel('Cycle duration (%)')
        matplotlib.rc('ytick', labelsize=6)
        plt.tight_layout()