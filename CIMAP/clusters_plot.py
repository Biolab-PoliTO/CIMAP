from .get_target_graph import get_target_graph
from .intervals import intervals

import warnings
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


__all__ = ['clusters_plot']

def clusters_plot(cimap_output, target = 'All', color = False):
    """
    Method for the visualization of the clustering results of CIMAP.

    :Input: * **cimap_output** (*dict*): the dictionary obtained as output from the cimap_outputput function.
            * **target**: "All" (*default*), labels or portion of the label of the muscle that you want to plot. For example if LGS_R and LGS_L are available inserting the target "LGS" would plot both. Inserting the exact label give as output the graphs related to that label.
            * **color** (*boolean*): False (*default*), parameter to set the color of the clusters matching."""
    
    # checking that the input is correct

    keys = list(cimap_output.keys())
    if "name" not in keys:
        raise ValueError('"name" key of "cimap_output" not found, check "cimap_output" dictionary'
                         'be sure to run cimap_output first')
    if "clusters" not in keys:
        raise ValueError('"clusters" key of "cimap_output" not found, check "cimap_output" dictionary'
                         'be sure to run cimap_output first')
    if "non_significant" not in keys:
        raise ValueError('"non_significant" key of "cimap_output" not found, check "cimap_output" dictionary'
                         'be sure to run cimap_output first')
    
    to_plot = get_target_graph(cimap_output,target)


    
    # one muscle at a time
    for muscle, muscle_name in enumerate(to_plot["name"]):
        count = 0
        # one modality at a time
        for _, cluster in enumerate(to_plot["clusters"][muscle]):
            if bool(cluster):
                count += 1
        if to_plot["non_significant"][muscle][0].any():
            count += 1
        
        # create subplot for each modality
        if count == 1:
            fig, axes = plt.subplots(count, squeeze=False)
            axes = axes[0]
        else:
            fig, axes = plt.subplots(1, count, figsize=(12, 7.5))

        count = 0
        for modality, cluster in enumerate(to_plot["clusters"][muscle]):
            counter = 0
            # extracting clusters one at a time
            if bool(cluster):
                X = np.linspace(0, 100, cluster[0].shape[1])
                ticks, labels = [], []

                for k in range(1, max(cluster[1]) + 1):
                    mask = cluster[1] == k
                    # if the cluster exists and is not a single element
                    if len(np.where(mask)[0]) > 1:
                        elements = cluster[0][mask]
                        color_in = cluster[2][mask] if color else 'b'
                        idx = cluster[3][mask]

                        # calculating the prototype
                        out, _, _ = intervals(elements)
                        proto = np.zeros((cluster[0].shape[1]))
                        ins = np.median(out, axis=0) * proto.shape[0] / 100
                        ins -= 1
                        elements[elements == 0] = np.nan

                        # plotting the cluster
                        for jj, act in enumerate(elements):
                            counter -= 1
                            axes[count].plot(X, act * counter, str(color_in[0]), linewidth=2)
                            ticks.append(counter)
                            labels.append(str(int(idx[jj])))

                        # marking the prototype intervals
                        for p in range(len(ins)):
                            if p % 2 == 0:
                                proto[int(ins[p]):int(ins[p + 1]) + 1] = 1
                        proto[proto == 0] = np.nan
                        counter -= 1
                        axes[count].plot(X, proto * counter, 'k', linewidth=4)
                        ticks.append(counter)
                        labels.append('P' + str(int(k)))
                        counter -= 1

                    # if it is a single element cluster, plot it thicker like a prototype
                    elif len(np.where(mask)[0]) == 1:
                        element = cluster[0][mask]
                        element[element == 0] = np.nan
                        idx = cluster[3][mask]
                        counter -= 1
                        color_in = cluster[2][mask] if color else 'b'
                        axes[count].plot(X, element.reshape(-1) * counter, str(color_in[0]), linewidth=4)
                        ticks.append(counter)
                        labels.append('P' + str(int(k)) + ' - ' + str(int(idx[0])))
                        counter -= 1

                # setting the plot parameters
                axes[count].set_yticks(ticks)
                axes[count].set_yticklabels(labels, fontsize=8)
                axes[count].set_ylim((counter - 2, 2))
                axes[count].set_xlim((0, 100))
                axes[count].grid(alpha=0.6)
                axes[count].set_title(f"{modality} modality" if modality == 1 else f"{modality} modalities")
                count += 1

        # plotting the non_significant modalities
        if to_plot["non_significant"][muscle][0].any():
            ns = to_plot["non_significant"][muscle][0].copy()
            idx = to_plot["non_significant"][muscle][1]
            counter = 0
            ticks, labels = [], []
            _, nact, _ = intervals(ns)

            for ii, ac in enumerate(ns):
                counter -= 1
                ac[ac == 0] = np.nan
                axes[count].plot(X, ac * counter, 'b', linewidth=2)
                ticks.append(counter)
                labels.append(f"{int(nact[ii])} mod - {int(idx[ii])}")

            # setting plot parameters for non_significant data
            axes[count].set_yticks(ticks)
            axes[count].set_yticklabels(labels, fontsize=8)
            axes[count].set_ylim((counter - 2, 2))
            axes[count].set_xlim((0, 100))
            axes[count].set_title("Modalities under Th = 10")
            axes[count].grid(alpha=0.6)

        fig.suptitle(muscle_name, fontsize=15)
        fig.supxlabel('Cycle duration (%)', fontsize=15)
        fig.supylabel("Gait cycles", fontsize=15)
        fig.tight_layout(pad=1)