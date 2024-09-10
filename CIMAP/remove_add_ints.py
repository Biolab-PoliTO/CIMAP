import numpy as np
import warnings
from .intervals import intervals

warnings.simplefilter("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore", category=UserWarning)

__all__ = ['remove_add_ints']

def remove_add_ints(s):
    """
    Function for the identification and removal of full active, full non-active cycles as this cycles are identified as outliers. The function also removes the activation intervals and small gaps that are smaller than 3% of the cycles as it was demonstrated in  `[1]`_ that such length in the activation is ininfluent on the biomechanics of the task and so they represent noise.

    .. _[1]: https://doi.org/10.1109/10.661154
    
    :Input: * **s** (*dict*): the dictionary containing the cycles activation data created by the Input function from the *.csv* file.

    :Output: * **s** (*dict*): same as the dictionary in input modified with the removal of the small gaps and of the always on/off cycles.
    """
    # check that the field "Cycles" exists and is corectly labeled 
    keys = list(s.keys())
    keys_l = [k.lower() for k in keys]
    if "cycles" not in keys_l:
        raise ValueError('"Cycles" key of dictionary not found, check the input dictionary "s"')
    elif "Cycles" not in keys and "cycles" in keys_l:
        s["Cycles"] = s.pop("cycles")
        warnings.warn('"Cycles" field format wrong, corrected')
        
        
    # check for the correct format of the input variable
    for cyc in s["Cycles"]:
        if not(isinstance(s["Cycles"][0],np.ndarray)):
            raise ValueError('Wrong cycles format, must be a numpy array')
        if len(cyc.shape) != 2:
            raise ValueError('Wrong cycles format, must be a numpy array of 2 dimensions')
        if cyc.shape[1] != 1000:
            raise ValueError('Wrong cycles format, must be normalized to 1000 samples')
    
    smallValue = 3
    for f,cyc in enumerate(s["Cycles"]):
        # extraction of the activation intervals for each cycle
        cycles,nact,_ = intervals(cyc)
        
        # removal of the always OFF cycles
        if np.count_nonzero(nact==0):
            print("Full off cycles removed for muscle " +s["Labels"][f] + ": %s" % np.count_nonzero(nact==0))
            cycles = np.delete(cycles,nact==0,axis = 0)
            nact = np.delete(nact,nact==0,axis = 0)
        
    
        # process of the identification of the always ON cycles and removal
        # of the small gaps and activation intervals
        fullon = []
        for i,c in enumerate(cycles):
            ints = np.diff(c)
            # identification of the small gaps and intervals
            # position of the gaps/ints to remove
            rem = ints<=smallValue # 3% defined as the value under which an activation is considered non-sufficient for motion purpose
            # gap of interval that is placed ant the star or at the end of the cycle
            # is never removed because the information from the previous cycle is not given
            
            if c[0] == 0.1:
                rem[0] = False
            
            if c[-1] == 100:
                rem[-1] = False
            # iterative removal checking whether the removal creates new removable gaps/ints    
            while any(rem):
                idx = []
                for j in range(rem.size):
                    if rem[j] and j%2 != 0:
                        idx.append([j,j+1])
                c = np.delete(c,idx)
                ints = np.diff(c)
                rem = ints<=smallValue
                if c[0] == 0.1:
                    rem[0] = False
                
                if c[-1] == 100:
                    rem[-1] = False
                idx = []
                for j in range(rem.size):
                    if rem[j] and j%2 == 0:
                        idx.append([j,j+1])
                c = np.delete(c,idx)
            # check and save of the always ON cycles, performed later because
            # the smallValue removal can cause the creation of always on cycles
            ints = np.diff(c)
            if np.sum(ints[0::2])>99:
                fullon.append(i)
            cycles[i] = c
            nact[i] = len(c)/2
        # removal of always ON cycles

        if bool(fullon):
            cycles = np.delete(cycles,fullon,axis = 0)
            nact = np.delete(nact,fullon,axis = 0)

        if bool(fullon):
            print("Full on cycles removed for muscle " +s["Labels"][f] + ": %s" % len(fullon))
        # Recreation of the cycles as binary arrays to be stored in s["Cycles] again
        cyc_out = np.zeros((cycles.shape[0],cyc.shape[1]))
        for k,ins in enumerate(cycles):
            ins = ins*cyc_out.shape[1]/100
            ins -=1
            for p in range(len(ins)):
                if p%2 == 0:
                    cyc_out[k,int(ins[p]):int(ins[p+1])+1] = 1
                    
        s["Cycles"][f] = cyc_out
    print("Pre-processing successfully performed")
    return s