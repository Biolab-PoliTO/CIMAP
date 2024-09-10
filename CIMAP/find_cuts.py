import numpy as np
from .utils import smooth

__all__ = ['find_cuts']

def find_cuts(distance):
    '''Function for the automatic identification of the cutting point on the dendrograms `[1]`_.

    :Input: * **distance** (*numpyarray*): the distance array that is the third column of the linkage function output.

    :Output: * **cuts** (*list*): list containing the optimal cutting point for each type of cut used.'''
    dist_diff = np.diff(distance)
    mean = np.mean(dist_diff[round(0.5*len(dist_diff))-1:])
    std = np.std(dist_diff[round(0.5*len(dist_diff))-1:], ddof=1)
    
    x = smooth(dist_diff,5)
    i_sm = len(x)-1
    while x[i_sm]> x[i_sm-1] and i_sm > 0.8*len(x):
        if x[i_sm]> x[i_sm-1]:
            i_sm-=1
            
    
    idx3 = i_sm+2
    if any(list(dist_diff > mean)):
        idx1 = list(dist_diff > mean).index(True)+1
    else:
        idx1 = len(dist_diff)+1
    if any(list(dist_diff > mean+std)):
        idx2 = list(dist_diff > mean+std).index(True)+1
    else:
        idx2 = len(dist_diff)+1
    
    cuts =[idx1,idx2,idx3]
    return cuts