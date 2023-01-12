import seaborn as sns
from scipy.cluster import hierarchy
import numpy as np

__all__ = ['algorithm_output']

def algorithm_output(s,muscles):
    '''Function for the creation of the output of the algorithm. The output structure of this function is used for the clusterplot graphical function for the representation of the results of clustering.
    
    :Input: * **s** (*dict*): the dictionary containing the cycles activation data created by the Input function from the *.csv* file.
            * **muscles** (*dict*): output dictionary from the "CIMAP_cuts" function containing the results of the clustering on the ciclical activation intervals
    :Output: * **cimap_out** (*dict*): dictionary that contains the results of clustering divided for each individual muscle given as input with the removal of the non significant cluster'''

    labels = []
    clustering = []
    non_significant = []
    
    # check for "clusters" field so that the input is correct
    keys = list(muscles.keys())
    if "clusters" not in keys:
        raise ValueError('"clusters" key of muscles not found, check "muscles" dictionary'
                         'be sure to run CIMAP_cuts first')
    # separation of each muscle one side at a time
    for i,lbl in enumerate(muscles["name"]):
        for k,sd in enumerate(muscles["side"][i]):
            clusters = []
            ns = []
            # flag for the first creation of the non significant np.array
            flag = 0
            if sd == 0:
                lb = lbl+"_L"
            else:
                lb = lbl+"_R"
            for j,clus in enumerate(muscles["clusters"][i]):
                
                if any(clus):
                    # identification of the cycles to extract
                   ext = muscles["modalities"][i][j][:,-1] == sd
                   # number of clusters
                   ncl = muscles['clusters'][i][j].max()
                   # threshold for the coloring of the dendrogram
                   t2 = muscles['dendrograms'][i][j][-1-(ncl-1),2]
                   t1 = muscles['dendrograms'][i][j][-1-(ncl-2),2]
                   # setting the color palette
                   link_color_pal = sns.color_palette("tab10")
                   link_color_pal = list(link_color_pal.as_hex())
                   # removing orange and red from the color palette because of 
                   # later graphical functions
                   link_color_pal.pop(3)
                   link_color_pal.pop(1)
                   hierarchy.set_link_color_palette(link_color_pal)
                   # construction of the dendrograms and extraction of the colors
                   # associated with the clusters
                   t = (t1+t2)/2
                   dn = hierarchy.dendrogram(muscles['dendrograms'][i][j],color_threshold = t,no_plot=True)
                   cols = np.array([x for _,x in sorted(zip(dn['leaves'],dn['leaves_color_list']))])
                   # adding the clusters to the list of clusters for the muscle
                   # taking them from the original data and not with different
                   # number of columns given by the modality
                   # Also the information about the cluster that the cycle belong to,
                   # the temporal sequence and color are stored
               
                   clusters.append([s["Cycles"][muscles["pos"][i][k]][muscles["modalities"][i][j][ext,-2].astype(int)-1,:],clus[ext],cols[ext],muscles["modalities"][i][j][ext,-2]])
                elif muscles["modalities"][i][j].any() and not(any(clus)):
                    if flag == 0:
                        # extraction of the non significative clusters
                        ext = muscles["modalities"][i][j][:,-1] == sd
                        ns = s["Cycles"][muscles["pos"][i][k]][muscles["modalities"][i][j][ext,-2].astype(int)-1,:]
                        if ns.any():
                            # to avoid situation when non of the cycles is non significant
                            # to have an empty list
                            ns_idx = muscles["modalities"][i][j][ext,-2]
                        else:
                            ns_idx = []
                        flag = 1
                        clusters.append([])
                    else:
                        ext = muscles["modalities"][i][j][:,-1] == sd
                        ns_ins = s["Cycles"][muscles["pos"][i][k]][muscles["modalities"][i][j][ext,-2].astype(int)-1,:]
                        if ns_ins.any():
                            ns = np.vstack((ns,ns_ins))
                            ns_ins = muscles["modalities"][i][j][ext,-2]
                        else:
                            ns_ins = []
                        ns_idx = np.hstack((ns_idx,ns_ins))
                        clusters.append([])
                else:
                    clusters.append([])

            clustering.append(clusters)
            if flag == 1:
             non_significant.append([ns,ns_idx])
            else:
                non_significant.append(np.array([ns,[]]))
            labels.append(lb)
    cimap_out = {
           "name": labels,
            "clusters": clustering,
            "non_significant":non_significant        
       }
    print("Output dictionary created")
    return cimap_out