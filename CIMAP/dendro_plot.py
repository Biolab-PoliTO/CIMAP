from .get_target_graph import get_target_graph

import warnings
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster import hierarchy

__all__ = ['dendro_plot']

def dendro_plot(muscles,target = 'All'):
    """
    Method for plotting the dendrograms built with CIMAP and chosen after performing the clustering process.

    :Input: * **muscles** (*dict*): the dictionary obtained as output from the CIMAP_cuts function.
            * **target**: "All" (*default*), labels or portion of the label of the muscle that you want to plot. For example if LGS_R and LGS_L are available inserting the target "LGS" would plot both. Inserting the exact label give as output the graphs related to that label.

    """
    # check for "dendrograms" and "clusters" field so that the input is correct
    keys = list(muscles.keys())
    if "dendrograms" not in keys:
        raise ValueError('"dendrograms" key of muscles not found, check "muscles" dictionary'
                         'be sure to run CIMAP.dendrograms and CIMAP.cuts first')
    if "clusters" not in keys:
        raise ValueError('"clusters" key of muscles not found, check "muscles" dictionary'
                         'be sure to run CIMAP.cuts first')
    
    to_plot = get_target_graph(muscles,target)
    for i,dens in enumerate(to_plot['dendrograms']):
        count = 0
        for j,dendro in enumerate(dens):  
            if len(dendro):
                count +=1
        if count == 1:
            fig,axes = plt.subplots(count,squeeze = False)
            axes = axes[0]
        else:
            fig,axes = plt.subplots(count,1)
 
        count = 0
        for j,dendro in enumerate(dens):
            if len(dendro):
                
                # retrieving the number of clusters
                ncl = to_plot['clusters'][i][j].max()
                # calculating the threshold for coloring the clusters
                t2 = dendro[-1-(ncl-1),2]
                t1 = dendro[-1-(ncl-2),2]
                t = (t1+t2)/2
                # setting the color palette
                link_color_pal = sns.color_palette("tab10")
                link_color_pal = list(link_color_pal.as_hex())
                link_color_pal.pop(3)
                link_color_pal.pop(1)
                hierarchy.set_link_color_palette(link_color_pal)
                # building the dendrogram graphically
                hierarchy.dendrogram(dendro,ax = axes[count],color_threshold = t, above_threshold_color='k')
                
                if to_plot['metrics'][i][j][0] == 1:
                    cut = 'CutA'
                elif to_plot['metrics'][i][j][0] == 2:
                    cut = 'CutB'
                elif to_plot['metrics'][i][j][0] == 3:
                    cut = 'CutB'
                if j == 1:
                    tit = str(j)+ ' modality - ' + cut + ' - metric: ' + to_plot['metrics'][i][j][1] 
                else:
                    tit = str(j)+ ' modalities - ' + cut + ' - metric: ' + to_plot['metrics'][i][j][1] 
                axes[count].set_title(tit)
                count +=1
                plt.yticks(fontsize = 8)
                plt.xticks(fontsize = 8)
        fig.supxlabel('Cycles',fontsize=15)
        fig.supylabel('Distance',fontsize=15)
        fig.suptitle(to_plot['name'][i],fontsize=15,x = 0.1, y=.95)
        fig.tight_layout(pad = 0.5)