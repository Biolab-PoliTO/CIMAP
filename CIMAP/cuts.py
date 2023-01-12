from .findcuts import findcuts

import numpy as np
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import pdist
from scipy.spatial import distance_matrix
import warnings


warnings.simplefilter("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore", category=np.VisibleDeprecationWarning)

__all__ = ['cuts']

def cuts(muscles):
    '''Function for the identification of the optimal cut among the three different automatic cuts performed on the dendrograms based on distance `[2]`_.

    :Input: * **muscles** (*dict*): the dictionary containing the cycles divided by modality, and the dendrgrams got as output from the CIMAP_dendrograms function.

    :Output: * **muscles** (*dict*): dictionary containing the best clustering obtained for each muscle and modality.'''
    # check for "dendrograms" field so that the input is correct
    keys = list(muscles.keys())
    if "dendrograms" not in keys:
        raise ValueError('"dendrograms" key of muscles not found, check "muscles" dictionary'
                         'be sure to run CIMAP_dendrograms first')
    # lists that will be added to the dictionary
    dendrograms_struct = []
    clusters_struct = []
    metrics_struct = []
    # configuration value assignmement
    
    for i,den in enumerate(muscles["dendrograms"]):
        # list of clusters an dendrograms
        clusters = []
        dendrograms = []
        norm = []
        for j,ds in enumerate(den):
            clust = [];
            if bool(ds):
                cut_ind = np.zeros((2,1))-1
                chosen = np.zeros((2,1))-1
                for jj,dd in enumerate(ds):
                    # extraction of the three cuts using the function find cuts
                    cuts = findcuts(dd[:,2])

                    for c in range(len(cuts)):
                        # clustering from the cuts
                        cut = fcluster(dd, np.array(muscles["modalities"][i][j].shape[0])-cuts[c]
                                          ,criterion = 'maxclust')
                        # identification of single element clusters and calculation of the mean
                        # distances of the elements inside the clusters
                        single = 0
                        mean_dist = []
                        for v in set(cut):
                            l = [k for k,z in enumerate(cut) if z == v]
                            mean_dist.append(np.mean(pdist(muscles["modalities"][i][j][l,0:-2],metric='cityblock')))
                            if len(l) == 1:
                                single+=1
                            

                        
                        mean_dist = np.delete(mean_dist,np.isnan(mean_dist))
                        # the first iteration is always tken in consideration
                        if cut_ind[jj] == -1:
                            # the cut index is (intra_var*n° of meaningful clusters)/n° of elements in the meaningful clusters
                            # the former cut index privileged too much cuts with higher numbers of small clusters
                            cut_ind[jj] = (np.sum(mean_dist)*len(mean_dist))/(muscles["modalities"][i][j].shape[0]-single)
                            clust.append(cut)
                            chosen[jj] = c+1
                        else:
                            if (np.sum(mean_dist)*len(mean_dist))/(muscles["modalities"][i][j].shape[0]-single) < cut_ind[jj]:
                                cut_ind[jj] = (np.sum(mean_dist)*len(mean_dist))/(muscles["modalities"][i][j].shape[0]-single)
                                clust[jj] = cut
                                chosen[jj] = c+1
                
                # evaluation of the best metrics between the two possible if used
                # using clust variability
                clust_var = []        
                for cl in clust:
                     z = [np.mean(distance_matrix(muscles["modalities"][i][j][cl == v_L,0:-2],np.reshape(np.median(muscles["modalities"][i][j][cl == v_L,0:-2],axis=0),(1,-1)),p=1)) for v_L in set(cl)]
                     z =  np.array(z)
                     dim = np.array([len(np.where(cl == v_L)[0]) for v_L in set(cl)])
                     
                     clust_var.append(np.mean(np.delete(z,np.where(dim==1)[0])))
                idx = np.where(clust_var == min(clust_var))[0][0]
                if idx == 0:
                    norm.append([chosen[idx], 'L1 norm'])

                else:
                    norm.append([chosen[idx], 'L inf norm'])


                clusters.append(clust[idx])
                dendrograms.append(muscles["dendrograms"][i][j][idx])

                        
            else:
               dendrograms.append([])
               clusters.append([])
               norm.append([])
                
        dendrograms_struct.append(dendrograms)  
        clusters_struct.append(clusters)
        metrics_struct.append(norm)
    muscles["dendrograms"] = dendrograms_struct
    muscles["clusters"] = clusters_struct
    muscles["metrics"] = metrics_struct
    print("Best clustering result chosen")
    return muscles