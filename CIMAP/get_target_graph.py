__all__ = ['get_target_graph']

def get_target_graph(target_dict, target):
    ''' Utility method that allow the user to decide what muscle given as input output graphically.

    :Input: * **targetdict** (*dict*): any dictionary given as input or obtained as output from CIMAP functions from which the user want to plot a muscle.
    
    :Returns: * **outputdict** (*dict*): dictionary containing the information necessary for the graphical function to plot the desired muscle.
    '''
    output_dict = {}
    
    if target.lower() == "all":
        output_dict = target_dict.copy()
    else:
        keys = list(target_dict.keys())
        keys_lower = [key.lower() for key in keys]

        if "labels" in keys_lower:
            index = [i for i, key in enumerate(keys_lower) if key == "labels"]
            name = keys[index[0]]
        elif "name" in keys_lower:
            index = [i for i, key in enumerate(keys_lower) if key == "name"]
            name = keys[index[0]]
        else: 
            raise ValueError("Invalid key format in the dictionaries.")
        
        index = [i for i, label in enumerate(target_dict[name]) if target in label]
        for key in keys:
            if key.lower() == 'subject':
                output_dict[key] = target_dict[key]
            else:
                output_dict[key] = [target_dict[key][i] for i in index]
    
    return output_dict