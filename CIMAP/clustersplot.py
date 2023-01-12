from .targetgraph import targetgraph
from .intervals import intervals

import warnings
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


__all__ = ['clustersplot']

def clustersplot(cimap_out,target = 'All',color = False):
    """
    Method for the visualization of the clustering results of CIMAP.

    :Input: * **cimap_out** (*dict*): the dictionary obtained as output from the CIMAP_output function.
            * **target**: "All" (*default*), labels or portion of the label of the muscle that you want to plot. For example if LGS_R and LGS_L are available inserting the target "LGS" would plot both. Inserting the exact label give as output the graphs related to that label.
            * **color** (*boolean*): False (*default*), parameter to set the color of the clusters matching."""
    
    # checking that the input is correct

    keys = list(cimap_out.keys())
    if "name" not in keys:
        raise ValueError('"name" key of "cimap_out" not found, check "cimap_out" dictionary'
                         'be sure to run CIMAP_output first')
    if "clusters" not in keys:
        raise ValueError('"clusters" key of "cimap_out" not found, check "cimap_out" dictionary'
                         'be sure to run CIMAP_output first')
    if "non_significant" not in keys:
        raise ValueError('"non_significant" key of "cimap_out" not found, check "cimap_out" dictionary'
                         'be sure to run CIMAP_output first')
    
    toplot = targetgraph(cimap_out,target)
    # one muscle at a time
    for i,mslname in enumerate(toplot["name"]):
        # one modality at a time
        cont = 0
        for j,clus in enumerate(toplot["clusters"][i]):
            if bool(clus):
                cont+=1
        if toplot["non_significant"][i][0].any():
            cont+=1
        if cont == 1:
            fig,axes = plt.subplots(cont,squeeze = False)
            axes = axes[0]
        else:
            fig,axes = plt.subplots(1,cont,figsize=(12,7.5))
        
        cont = 0
        for j,clus in enumerate(toplot["clusters"][i]):
            counter = 0
            # extracting clusters one at a time
            if bool(clus):
                
                X = np.linspace(0,100,clus[0].shape[1])
                tick = []
                label = []
                
                for k in range(1,max(clus[1])+1):
                    ext = clus[1] == k
                    # if the cluster exists and is not a single element
                    if len(np.where(ext)[0])>1:
                        el = clus[0][ext]
                        # get the color or it is set to blue
                        if color:
                            color_in = clus[2][ext]
                        else:
                            color_in = 'b'
                        idx = clus[3][ext]
                        # calculating the prototype
                        out,_,_ = intervals(el)
                        proto = np.zeros((clus[0].shape[1]))
                        ins = np.median(out,axis = 0)*proto.shape[0]/100
                        ins -=1
                        el[el==0] = np.nan
                        # plotting the cluster
                        for jj,act in enumerate(el):
                            counter -=1
                            axes[cont].plot(X,act*counter, str(color_in[0]), linewidth = 2)
                            tick.append(counter)
                            label.append(str(int(idx[jj])))
                        for p in range(len(ins)):
                            if p%2 == 0:
                                proto[int(ins[p]):int(ins[p+1])+1] = 1
                        proto[proto == 0] = np.nan
                        counter -=1
                        axes[cont].plot(X,proto*counter,'k',linewidth=4)
                        tick.append(counter)
                        label.append('P'+str(int(k)))
                        counter -=1
                    # if is a single element cluster is plotted thicker like a
                    # prototype
                    if len(np.where(ext)[0]) == 1:
                        el = clus[0][ext]
                        el[el==0] = np.nan
                        idx = clus[3][ext]
                        counter -=1
                        if color:
                            color_in = clus[2][ext]
                        else:
                            color_in = 'b'
                        axes[cont].plot(X,np.reshape(el,(-1,))*counter,str(color_in[0]),linewidth=4)
                        tick.append(counter)
                        label.append('P' + str(int(k))+' - ' + str(int(idx[0])))
                        counter -=1
                axes[cont].set_yticks(tick) 
                axes[cont].set_yticklabels(label, fontsize = 8)
                axes[cont].set_ylim((counter-2,2))
                axes[cont].set_xlim((0,100))
                axes[cont].grid(alpha = 0.6)
                if j == 1:
                    axes[cont].set_title(str(j)+" modality")
                else:
                    axes[cont].set_title(str(j)+" modalities")

                cont +=1

        if toplot["non_significant"][i][0].any():
            # plotting the non_significant modalities with the reference to the
            # number of modality each cycles has

            ns = toplot["non_significant"][i][0].copy()
            idx = toplot["non_significant"][i][1]
            counter = 0
            tick = []
            label = []
            _,nact,_ = intervals(ns)
             #ns[ns==0] = np.nan
            for ii,ac in enumerate(ns):
                 counter-=1
                 ac[ac==0] = np.nan
                 axes[cont].plot(X,ac*counter,'b',linewidth = 2)
                 tick.append(counter)
                 label.append(str(int(nact[ii]))+' mod - '+ str(int(idx[ii])))
            axes[cont].set_yticks(tick) 
            axes[cont].set_yticklabels(label, fontsize = 8)
            axes[cont].set_ylim((counter-2,2))
            axes[cont].set_xlim((0,100))
            axes[cont].set_title("Modalities under Th = 10")
            axes[cont].grid(alpha = 0.6)
        fig.suptitle(mslname,fontsize=15)
        fig.supxlabel('Cycle duration (%)',fontsize=15)
        fig.supylabel("Gait cycles",fontsize=15)
        fig.tight_layout(pad = 1)