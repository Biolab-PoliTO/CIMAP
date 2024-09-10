import numpy as np
from .intervals import intervals

__all__ = ['modality_division']

def modality_division(s,muscles):
    """Function for the division of the gait cycles in the different modalities before clustering. The function uses the function intervals to retrieve the activation intervals and then it divides them inside a list in muscles where the index identifies the number of activation intervals of the modality (0 always empty).
    
    :Input: * **s** (*dict*): the dictionary containing the cycles activation data created by the Input function from the *.csv* file.
            * **muscles** (*dict*): dictionary obtained as output of CIMAP_input.
    :Output: * **muscles** (*dict*): dictionary that contains the cycles divided by modality represented by the value of edges of the activation intervals translated as percentage of the cycle values."""
    # check for "side" and "pos" fields so that the input is correct
    keys = list(muscles.keys())
    if "side" not in keys:
        raise ValueError('"side" key of muscles not found, check "muscles" dictionary'
                         'be sure to run CIMAP_input first')
    if "pos" not in keys:
        raise ValueError('"pos" key of muscles not found, check "muscles" dictionary'
                         'be sure to run CIMAP_input first')    
    # retrieve of the information to get the cycles
    side = muscles["side"]
    pos = muscles["pos"]
    modalities = []
    for i,p in enumerate(pos):
        # extraction of the cycles
        cyc = [s["Cycles"][i] for i in p]
        # flag for the first time is performed
        flag = 0
        for k,c in enumerate(cyc):
            # calculation of the number of activation intervals and of the
            # intervals ends
             inters,nact,idx = intervals(c)
             sd = np.zeros(len(nact))+side[i][k]

             if flag == 0:
                intr = list(inters)
                acts = np.vstack((idx,sd,nact)).T
                flag = 1
             else:
                intr = intr+list(inters)
                a_ins = np.vstack((idx,sd,nact)).T
                acts = np.vstack((acts,a_ins))

        mods = []
        # sorting of the modalities
        for n in range(int(max(acts[:,-1])+1)):
            if any(acts[:,-1]==n):
                ins = np.array([])
                flag = 0
                for k,_ in enumerate(acts):
                    if acts[k,-1]==n:
                        if flag == 0:
                            ins = intr[k].T
                            flag = 1
                        else:
                            ins = np.vstack((ins,intr[k]))
                if len(np.where(acts[:,-1]==n)[0]) == 1:
                    ins =np.append(ins,acts[acts[:,-1]==n,:-1])
                    ins = np.reshape(ins,(1,-1))
                else:
                    ins = np.hstack((ins,acts[acts[:,-1]==n,:-1]))
            else:
                ins = np.array([])
            mods.append(np.array(ins, dtype=np.float64))
        modalities.append(mods)
    muscles["modalities"] = modalities
    print("Cycles successfully divided into modalities")
    return muscles