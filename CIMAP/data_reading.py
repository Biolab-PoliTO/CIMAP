import warnings
import numpy as np
from .utils import csv2dict
import csv

warnings.simplefilter("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore", category=np.VisibleDeprecationWarning)

__all__ = ['data_reading']

def data_reading(input_file):
    '''Function that takes the input of CIMAP and prepares the data structures for the application of CIMAP Algorithm.
    
    :Input: * **input_file** (*string*): a string containing the path of the *.csv* file that contains the input data for the application of CIMAP. Be sure that the input data respects the requirements set in the Data Requirements section of the documentation.
    
    :Output: * **s** (*dict*): data structure of CIMAP that is used for the application of the methods of the algorithm.
             * **muscles** (*dict*): dictionary containing the information about the data that is given as input to CIMAP. Specifically the muscles, if the acquisition is bilateral and the position where to retrieve the cycles.'''
    
    # Application of the ausiliary method that reads the data from the *.csv* file and transforms it into the dictionary that is used in the following methods.
    s = csv2dict(input_file)


    # check that the field "Cycles" exists and is corectly labeled
    keys = list(s.keys())
    keys_l = [k.lower() for k in keys]
    if not("cycles" in keys_l):
        raise ValueError('"Cycles" key of dictionary not found, check the input dictionary')
    elif not("Cycles" in keys) and "cycles" in keys_l:
        s["Cycles"] = s.pop("cycles")
        warnings.warn(' "Cycles" field format wrong, corrected')

        
    # check that the field "Labels" exists and is corectly labeled 
    if not("labels" in keys_l):
        raise ValueError('"Labels" key of dictionary not found, check the input dictionary')
    elif not("Labels" in keys) and "labels" in keys_l:
        s["Labels"] = s.pop("labels")
        warnings.warn(' "Labels" field format wrong, corrected')
    
    # check for the correct format of the input variable  
    for cyc in s["Cycles"]:
        if not(isinstance(s["Cycles"][0],np.ndarray)):
            raise ValueError('Wrong cycles format, must be a numpy array')
        if len(cyc.shape) != 2:
            raise ValueError('Wrong cycles format, must be an array of 2 dimensions')
        if cyc.shape[1] != 1000:
            raise ValueError('Wrong cycles format, must be normalized to 1000 samples')
    
    # extraction of the labels of the muscles acquired
    mslnames = [];
    side = np.empty((0))
    for lbl in s["Labels"]:
        if(lbl[-1]=="L"):
            side = np.append(side,np.array([0]),axis=0)
        elif(lbl[-1]=="R"):
            side = np.append(side,np.array([1]),axis=0)
        else:
            raise ValueError('Wrong label format')
        if lbl[-2] == '_':
            mslnames.append(lbl[:-2])
        else:
            mslnames.append(lbl[:-1])
    
    # identification of the position inside the labels list where the muscles are
    # positioned and which side or sides are in the input list. This is made for
    # following procedures inside CIMAP algorithm
    msl_list = set(mslnames)
    pos = [];
    names = []
    for j,x1 in enumerate(msl_list):
         pos.append([i for i,x in enumerate(mslnames) if x == x1])
         names.append(x1)
    side_out = []
    for c in pos:
        side_out.append(side[c])
        
    muscles = {
        "name": names,
        "side": side_out,
        "pos":pos        
    }
    print("Input dataset loaded successfully")
    return s,muscles

