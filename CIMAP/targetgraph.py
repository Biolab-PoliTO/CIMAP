__all__ = ['targetgraph']

def targetgraph(targetdict,target):
    ''' Utility method that allow the user to decide what muscle given as input output graphically.

    :Input: * **targetdict** (*dict*): any dictionary given as input or obtained as output from CIMAP functions from which the user want to plot a muscle.
    
    :Returns: * **outputdict** (*dict*): dictionary containing the information necessary for the graphical function to plot the desired muscle.
    '''
    outputdict = {}
    if target.lower() == "all":
        outputdict = targetdict.copy()
    else:
        keys = list(targetdict.keys())
        keys_l = [k.lower() for k in keys]
        if "labels" in keys_l:
            idx = [i for i,k in enumerate(keys_l) if k == "labels"]
            name = keys[idx[0]]
        elif "name" in keys_l:
            idx = [i for i,k in enumerate(keys_l) if k == "name"]
            name = keys[idx[0]]
        else: 
            raise ValueError("wrong name format in the dictionaries")
        
        idx = [i for i,label in enumerate(targetdict[name]) if target in label]
        for k in keys:
            if k.lower() == 'subject':
                outputdict[k] = targetdict[k]
            else:
                outputdict[k] = [targetdict[k][ind] for ind in idx]
    return outputdict