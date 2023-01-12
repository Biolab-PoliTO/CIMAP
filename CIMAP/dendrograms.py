from scipy.cluster.hierarchy import linkage

__all__ = ['dendrograms']

def dendrograms(muscles):
    """
    Function for building of the dendrograms with the L1 and L infinite metrics for the clustering process `[2]`_.

    .. _[2]: https://doi.org/10.1109/EMBC.2017.8036762

    :Input: * **muscles** (*dict*): the dictionary containing the cycles divided by modality got as output from the modality division function.
          
    :Output: * **muscles** (*dict*): dictionary containing the dendrograms built from the cycles divided in modalities"""
    
    
    # check for "modlaities" field so that the input is correct
    keys = list(muscles.keys())
    if "modalities" not in keys:
        raise ValueError('"modalities" key of muscles not found, check "muscles" dictionary'
                         'be sure to run modality_division first')
    # retrieve of configuration parameters
    dendros = []
    
    for msl in muscles["modalities"]:
        # calculation of the value of the % of cycle to be used as threshold

                
        dendro_mod = []
        for mod in msl:
            dendro = []
            if bool(mod.any()):
                # check if the threshold parameter is respected
                if mod.shape[0] >= 10:

                    # dendrogram for the L1 metric
                    dendro.append(linkage(mod[:,0:-2], method = 'complete', metric = 'cityblock'))

                    # dendrogram for the Linf metric    
                    dendro.append(linkage(mod[:,0:-2], method = 'complete', metric = 'chebyshev'))

            dendro_mod.append(dendro)
        dendros.append(dendro_mod)
     
    muscles["dendrograms"] = dendros
    print("Dendrograms building completed")
    return muscles       