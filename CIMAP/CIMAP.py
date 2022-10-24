import warnings, csv,os, tkinter, tkinter.filedialog
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from scipy.spatial import distance_matrix
import seaborn as sns
from scipy.cluster import hierarchy
import matplotlib
import matplotlib.pyplot as plt

warnings.simplefilter("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore", category=np.VisibleDeprecationWarning)

def Run(input_file = None):
    ''' Method for the application of CIMAP to a dataset. This method when used applies all the methods of the algorithm to the data in the *input_file*.

        :Input: * **input_file** (*string*): None (*default*), a string containing the path of the *.csv* file that contains the input data for the application of CIMAP. In case no input_file is given the system opens a window that allows the user to search and select manually the file to use as input.

        :Output: * **output_file**: the method automatically generates a *.csv* file in the same position of the input file containing the results of the application of the CIMAP. Refer to the **Data Requirements** section of the documentation for the detail on the output format of the data.
                 * **graphics**:  all the graphs related to the CIMAP Algorithm application are given as output to the user.
    '''
    # in case no input_file is given
    if not(input_file):
        print("Please choose the input file")
        # creation of the UI
        root = tkinter.Tk()
        root.attributes("-topmost", True)

        root.withdraw()

        input_file = tkinter.filedialog.askopenfilename(parent = root)
        root.destroy()
    # Reading of the data
    s,muscles = Input(input_file)
    # Removal of short intervals and fullon/fulloff cycles
    s = RemoveAddints(s)
    #division in modalities
    muscles = ModalityDivision(s,muscles)
    # construction of the dendrograms
    muscles = Dendrograms(muscles)
    # cut of the dendrogram and choice of the best clustering
    muscles = Cuts(muscles)
    # output of the CIMAP
    cimap_out = Output(s,muscles)
    # save of the output file
    _ = _outcsv(cimap_out,input_file)
    cimap_pas = PAsExtraction(cimap_out)
    print("CIMAP Algorithm application successfull")
    print("Graphical output generation")
    actplot(s)
    modality_distribution(s)
    dendroplot(muscles)
    clustersplot(cimap_out)
    PAsActivationsPlot(cimap_pas)
    plt.show(block = False)
    print("CIMAP graphical data produced")
    return

def Intervals(cycles):
     '''
     Method for the extraction of the activation intervals ends values. Also, the function returns the number of activation intervals in the cycle and the row where the cycle is put inside the "cycles" matrix. The row is used to mantain the sequence information of the cycles.

      :Input: * **cycles** (*numpyarray*): a numpy binary array whose rows represents the gait cycles and the columns represent the samples of the normalised cycle. It is important that the cycles are normalised all at the same value.

      :Return: * **out** (*list*): a list containing numpy arrays which contain the percentage value of the starting and ending point of the activation intervals (e.g., out[n] = [ON1,OFF1,...,ONn, OFFm])
             * **num** (*numpyarray*): a numpy array that contains the number of activation intervals of the activation interval stored in **out**
             * **idx** (*numpyarray*): a numpy array that contains the sequentail number that matches the cycles stored in **out**
         '''
      # check for the correct format of the input variable
        # check for the correct format of the input variable
    
     if not(isinstance(cycles,np.ndarray)):
         raise ValueError('Wrong cycles format, must be a numpy array')
     if len(cycles.shape) != 2:
         raise ValueError('Wrong cycles format, must be an array of 2 dimensions')
        
        # check whether the activation values are binary
     if np.multiply(cycles != (0),cycles != (1)).any():
         raise SystemExit('Wrong Activation values')
    
        # identificattion of the transitions
     gap = np.diff(cycles)
     out, num = [], []

     
     for j,g in enumerate(gap):
        # extration of the sample of the transition
         interval = [i for i,x in enumerate(g) if x!=0]
         if bool(interval):
            # if the first transition is -1 the activation starts at 0
            if g[interval[0]] == -1:
                interval.insert(0,0)
            # if the last transition is 1 the activation ends at 100
            if g[interval[-1]] == 1:
                interval.append(len(g))
            nact = len(interval)/2
         elif cycles[j,0] == 1:
            # always active cycle
            interval = [0, len(g)-1]
            nact = 1
         else:
            interval = []
            nact = 0
        # adding 1 to have the right percentage value
         for jj,n in enumerate(interval):
            if not(n == len(g)) and g[n] == 1:
                interval[jj] +=1
                

         num.append(nact)
         out.append(np.array(interval)+1) 
     out = np.array(out, dtype = object)*100/(len(g)+1) 
     num = np.array(num) 
     idx = np.arange(np.size(cycles,0))+1
     
     return out,num,idx

def Input(input_file):
    '''Method that takes the input of CIMAP and prepares the data structures for the application of CIMAP Algorithm.
    
    :Input: * **input_file** (*string*): a string containing the path of the *.csv* file that contains the input data for the application of CIMAP. Be sure that the input data respects the requirements set in the Data Requirements section of the documentation.
    
    :Return: * **s** (*dict*): data structure of CIMAP that is used for the application of the methods of the algorithm.
             * **muscles** (*dict*): dictionary containing the information about the data that is given as input to CIMAP. Specifically the muscles, if the acquisition is bilateral and the position where to retrieve the cycles.'''
    
    # Application of the ausiliary method that reads the data from the *.csv* file and transforms it into the dictionary that is used in the following methods.
    s = _csv2dict(input_file)


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
    return s,muscles 

def _csv2dict(input_file):
    ''' Ausiliary method that opens and reads the contents of the *.csv* file and rearranges it for the application of CIMAP '''
    labels,cycles = [],[]
    with open(input_file,'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            # checking that we are importing muscle data and not the header of the file
            if '_R' in row[0] or '_L' in row[0]:
                labels.append(row.pop(0))
                row = np.array(row).astype(float)
                # checking that all the activation values are 0 or 1 and not other values. NaNs are removed in case a different number of cycles is given as input from the tabular data
                row = row[np.isfinite(row)]
                if np.multiply(row != (0),row != (1)).any():
                    raise SystemExit('Wrong Activation values')

                if not(row.shape[0]/1000):
                    raise ValueError('csv input file has a wrong number of columns, check that the cycles are normalized to 1000 samples')
                cycles.append(row.reshape((int(row.shape[0]/1000),1000)))

    s = {
        "Labels":labels,
        "Cycles":cycles
    }
    file.close()
    return s

def RemoveAddints(s):
    """
    Method for the identification and removal of full active, full non-active cycles and short gaps or activation intervals.
    
    :Input: * **s** (*dict*): the dictionary containing the cycles activation data created by the Input function from the *.csv* file.

    :Returns: * **s** (*dict*): same as the dictionary in input modified with the removal of the small gaps and of the always on/off cycles.
    """
    # check that the field "Cycles" exists and is corectly labeled 
    keys = list(s.keys())
    keys_l = [k.lower() for k in keys]
    if "cycles" not in keys_l:
        raise ValueError('"Cycles" key of dictionary not found, check the input dictionary "s"')
    elif "Cycles" not in keys and "cycles" in keys_l:
        s["Cycles"] = s.pop("cycles")
        warnings.warn('"Cycles" field format wrong, corrected')
        
        
    # check for the correct format of the input variable
    for cyc in s["Cycles"]:
        if not(isinstance(s["Cycles"][0],np.ndarray)):
            raise ValueError('Wrong cycles format, must be a numpy array')
        if len(cyc.shape) != 2:
            raise ValueError('Wrong cycles format, must be a numpy array of 2 dimensions')
        if cyc.shape[1] != 1000:
            raise ValueError('Wrong cycles format, must be normalized to 1000 samples')
    
    smallValue = 3
    for f,cyc in enumerate(s["Cycles"]):
        # extraction of the activation intervals for each cycle
        cycles,nact,_ = Intervals(cyc)
        
        # removal of the always OFF cycles
        if np.count_nonzero(nact==0):
            print("Full off cycles removed: %s" % np.count_nonzero(nact==0))
        cycles = np.delete(cycles,nact==0)
        nact = np.delete(nact,nact==0)
        
    
        # process of the identification of the always ON cycles and removal
        # of the small gaps and activation intervals
        fullon = []
        for i,c in enumerate(cycles):
            ints = np.diff(c)
            # identification of the small gaps and intervals
            # position of the gaps/ints to remove
            rem = ints<=smallValue # 3% defined as the value under which an activation is considered non-sufficient for motion purpose
            # gap of interval that is placed ant the star or at the end of the cycle
            # is never removed because the information from the previous cycle is not given
            
            if c[0] == 0.1:
                rem[0] = False
            
            if c[-1] == 100:
                rem[-1] = False
            # iterative removal checking whether the removal creates new removable gaps/ints    
            while any(rem):
                idx = []
                for j in range(rem.size):
                    if rem[j] and j%2 != 0:
                        idx.append([j,j+1])
                c = np.delete(c,idx)
                ints = np.diff(c)
                rem = ints<=smallValue
                if c[0] == 0.1:
                    rem[0] = False
                
                if c[-1] == 100:
                    rem[-1] = False
                idx = []
                for j in range(rem.size):
                    if rem[j] and j%2 == 0:
                        idx.append([j,j+1])
                c = np.delete(c,idx)
            # check and save of the always ON cycles, performed later because
            # the smallValue removal can cause the creation of always on cycles
            ints = np.diff(c)
            if np.sum(ints[0::2])>99:
                fullon.append(i)
            cycles[i] = c
            nact[i] = len(c)/2
        # removal of always ON cycles

        cycles = np.delete(cycles,fullon)
        nact = np.delete(nact,fullon)

        if bool(fullon):
            print("Full on cycles removed: %s" % len(fullon))
        # Recreation of the cycles as binary arrays to be stored in s["Cycles] again
        cyc_out = np.zeros((cyc.shape))
        for k,ins in enumerate(cycles):
            ins = ins*cyc_out.shape[1]/100
            ins -=1
            for p in range(len(ins)):
                if p%2 == 0:
                    cyc_out[k,int(ins[p]):int(ins[p+1])+1] = 1
                    
        s["Cycles"][f] = cyc_out

    return s


def ModalityDivision(s,muscles):
    """Method for the division of the gait cycles in the different modalities before clustering. The function uses the function intervals to retrieve the activation intervals and then it divides them inside a list in muscles where the index identifies the number of activation intervals of the modality (0 always empty).
    
    :Input: * **s** (*dict*): the dictionary containing the cycles activation data created by the Input function from the *.csv* file.
            * **muscles** (*dict*): dictionary obtained as output of CIMAP_input.
    :Returns: * **muscles** (*dict*): dictionary that contains the cycles divided by modality represented by the value of edges of the activation intervals translated as percentage of the cycle values."""
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
             inters,nact,idx = Intervals(c)
             sd = np.zeros((inters.size))+side[i][k]

             if flag == 0:
                acts = np.vstack((inters,idx,sd,nact)).T
                flag = 1
             else:
                ins = np.vstack((inters,idx,sd,nact)).T
                acts = np.vstack((acts,ins))
        mods = []
        # sorting of the modalities
        for n in range(int(max(acts[:,3])+1)):
            if any(acts[:,3]==n):
                ins = np.vstack((acts[acts[:,3]==n,0]))
                ins = np.hstack((ins,acts[acts[:,3]==n,1:-1]))
            else:
                ins = np.array([])
            mods.append(np.array(ins, dtype=np.float))
        modalities.append(mods)
    muscles["modalities"] = modalities
    
    return muscles

def Dendrograms(muscles):
    """
    Method for building of the dendrograms with the L1 and L infinite metrics.
    The dendrogram is built using the scipy.cluster.hierarchy.linkage function. This function works similarly to the one in MATLAB but can give different results. The user can choose which metric to use by setting it in the "config.json" configuration file.

    :Input: * **muscles** (*dict*): the dictionary containing the cycles divided by modality got as output from the modality division function.
          
    :Return: * **muscles** (*dict*): dictionary containing the dendrograms built from the cycles divided in modalities"""
    
    
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
    
    return muscles       

def Cuts(muscles):
    '''Method for the identification of the optimal cut among the three different automatic cuts performed on the dendrograms based on distance.

    :Input: * **muscles** (*dict*): the dictionary containing the cycles divided by modality, and the dendrgrams got as output from the CIMAP_dendrograms function.

    :Return: * **muscles** (*dict*): dictionary containing the best clustering obtained for each muscle and modality.

    Check DOI: 10.1109/EMBC.2017.8036762 for more information'''
    # check for "dendrograms" field so that the input is correct
    keys = list(muscles.keys())
    if "dendrograms" not in keys:
        raise ValueError('"dendrograms" key of muscles not found, check "muscles" dictionary'
                         'be sure to run CIMAP_dendrograms first')
    # lists that will be added to the dictionary
    dendrograms_struct = []
    clusters_struct = []
    # configuration value assignmement
    
    for i,den in enumerate(muscles["dendrograms"]):
        # list of clusters an dendrograms
        clusters = []
        dendrograms = []
        for j,ds in enumerate(den):
            clust = [];
            if bool(ds):
                cut_ind = np.zeros((2,1))-1
                for jj,dd in enumerate(ds):
                    # extraction of the three cuts using the function find cuts
                    cuts = FindCuts(dd[:,2])

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
                            if len(l) == 1:
                                single+=1
                                mean_dist.append(np.mean(pdist(muscles["modalities"][i][j][l,0:-2],metric='cityblock')))
                            
                        mean_dist = np.delete(mean_dist,np.isnan(mean_dist))
                        # the first iteration is always tken in consideration
                        if cut_ind[jj] == -1:
                            # the cut index is (intra_var*n° of meaningful clusters)/n° of elements in the meaningful clusters
                            # the former cut index privileged too much cuts with higher numbers of small clusters
                            cut_ind[jj] = (np.sum(mean_dist)*len(mean_dist))/(muscles["modalities"][i][j].shape[0]-single)
                            clust.append(cut)
                        else:
                            if (np.sum(mean_dist)*len(mean_dist))/(muscles["modalities"][i][j].shape[0]-single) < cut_ind[jj]:
                                cut_ind[jj] = (np.sum(mean_dist)*len(mean_dist))/(muscles["modalities"][i][j].shape[0]-single)
                                clust[jj] = cut
                
                # evaluation of the best metrics between the two possible if used
                # using clust variability
                clust_var = []        
                for cl in clust:
                     z = [np.mean(distance_matrix(muscles["modalities"][i][j][cl == v_L,0:-2],np.reshape(np.median(muscles["modalities"][i][j][cl == v_L,0:-2],axis=0),(1,-1)),p=1)) for v_L in set(cl)]
                     z =  np.array(z)
                     clust_var.append(np.mean(z[z!=0]))
                idx = np.where(clust_var == min(clust_var))[0][0]
                

                clusters.append(clust[idx])
                dendrograms.append(muscles["dendrograms"][i][j][idx])

                        
            else:
               dendrograms.append([])
               clusters.append([])
                
        dendrograms_struct.append(dendrograms)  
        clusters_struct.append(clusters)
    muscles["dendrograms"] = dendrograms_struct
    muscles["clusters"] = clusters_struct
    return muscles

def FindCuts(distance):
    '''Method for the automatic identification of the cutting point on the dendrograms.

    :Input: * **distance** (*numpyarray*): the distance array that is the third column of the linkage function output.

    :Return: * **cuts** (*list*): list containing the optimal cutting point for each type of cut used.'''
    dist_diff = np.diff(distance)
    mean = np.mean(dist_diff[round(0.5*len(dist_diff))-1:])
    std = np.std(dist_diff[round(0.5*len(dist_diff))-1:], ddof=1)
    
    x = _smooth(dist_diff,5)
    i_sm = len(x)-1
    while x[i_sm]>= x[i_sm-1] and i_sm > 0.8*len(x):
        i_sm-=1
    
    idx3 = i_sm+1
    if any(list(dist_diff > mean)):
        idx1 = list(dist_diff > mean).index(True)+1
    else:
        idx1 = len(dist_diff)-1
    if any(list(dist_diff > mean+std)):
        idx2 = list(dist_diff > mean+std).index(True)+1
    else:
        idx2 = len(dist_diff)-1
    
    cuts =[idx1,idx2,idx3]
    return cuts


def _smooth(a,WSZ):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ    
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))

def Output(s,muscles):
    '''Method that reorders the results of the application of CIMAP to be easy to read and to be represented graphically.
    
    :Input: * **s** (*dict*): the dictionary containing the cycles activation data created by the Input function from the *.csv* file.
            * **muscles** (*dict*): output dictionary from the "CIMAP_cuts" function containing the results of the clustering on the ciclical activation intervals
    :Returns: * **cimap_out** (*dict*): dictionary that contains the results of clustering divided for each individual muscle given as input with the removal of the non significant cluster'''

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
                   t = muscles['dendrograms'][i][j][-1-(ncl-1),2]
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
                   dn = hierarchy.dendrogram(muscles['dendrograms'][i][j],color_threshold = t+0.05,no_plot=True)
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
            labels.append(lb)
    cimap_out = {
           "name": labels,
            "clusters": clustering,
            "non_significant":non_significant        
       }

    return cimap_out

def _outcsv(cimap_out,input_file, saving = True):
    ''' Ausiliary method for writing in a *.csv* file the results of the application of CIMAP.'''
    rows= []
    # for each muscle
    for i in range(len(cimap_out["name"])):
        
        row,position = [],[]
        # for each modality
        for j,cl in enumerate(cimap_out["clusters"][i]):
            if bool(cl):
                # creating the 6 digits code for each cycle
                row += list(map(lambda x: "{:0>2}".format(j)+"{:0>4}".format(x),cl[1]))
                position+=  cl[3].tolist()
        # checking the non significant cycles        
        if cimap_out["non_significant"][i][0].any():
            _,nact,_ = Intervals(cimap_out["non_significant"][i][0])
            # creating the 6 digits code for the non significant cycles
            row += list(map(lambda x: "{:0>2}".format(int(x))+"0000",nact))
            position += cimap_out["non_significant"][i][1].tolist()
        # rearraning to the sequential order of the cycles given in input
        row= [x for _,x in sorted(zip(position,row))]
        row.insert(0,cimap_out["name"][i])
        rows.append(row)
    # getting the path of the input file to write the file where the input_file is    
    if saving:
        ps = os.path.dirname(input_file)
        
        f = open(ps+"\\"+ os.path.splitext(os.path.basename(input_file))[0]+"_Output_CIMAP.csv", 'w')
        writer = csv.writer(f,lineterminator='\r')
        writer.writerows(rows)
        f.close()
    return rows

def PAsExtraction(cimap_out):
    '''Method for the extraction of the Principal Activations from the CIMAP results. The Principal Activations are those activation intervals, biomechanically speaking, strictly necessary for performing a task.
    
    :Input: * **cimap_out** (*dict*): the dictionary containing the results of the application of CIMAP.
    
    :Returns: * **cimap_pas** (*dict*): dictionary containing the result of the principal activations extraction. In the field "clusters" are contained the clusters used. In "non_significant" are stored the cycles removed using "Th_sig". In "prototypes" are stored the prototypes of the clusters, and in "PAs" are stored the principal activations.'''
    # retrieve of configuration parameter
    clusters = []
    non_significant = []
    prototypes = []
    principal_activations =[]
    colors = []
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

    for i in range(len(cimap_out["name"])):
        tot_cyc = 0
        single_el = 0
        clus = []
        cols = []
        # extraction of non significant cycles
        ns = cimap_out["non_significant"][i][0]
        for cl in cimap_out["clusters"][i]:
            if bool(cl):
                for j in range(1,max(cl[1])+1):
                    if any(cl[1] == j):
                        if cl[0][cl[1] == j,:].shape[0] == 1:
                            # counting the single element clusters after the division
                            # between sides
                            single_el += 1
                        # counting the total number of cycles
                        tot_cyc += cl[0][cl[1] == j,:].shape[0]
                        # extraction of the single clusters
                        clus.append(cl[0][cl[1] == j,:])
                        cols.append(cl[2][np.where(cl[1] == j)[0][0]])
        # calculation of the threshold value    
        th_sig = 0.1
            # single element clusters are not considered because they are already
            # considered as non significant
        sig = th_sig*(tot_cyc-single_el)
        if sig<=1:
            sig = 2

        
        dim = np.array([s.shape[0] for s in clus], dtype = object)
        
        to_move = dim<sig
        # in case all the cluster must be moved the biggest one is kept
        if to_move.all():
            to_move[dim == max(dim)] = False
        flag = 0  
        # calculation of the prototypes of the significant clusters
        for j,vl in enumerate(to_move):
            if vl:
                ns = np.vstack((ns,clus[j]))
            else:
                out,_,_ = Intervals(clus[j])
                
                proto = np.zeros((1,clus[j].shape[1]))
                ins = np.median(out,axis = 0)*proto.shape[1]/100
                ins -=1
                for p in range(len(ins)):
                    if p%2 == 0:
                        proto[0,int(ins[p]):int(ins[p+1])+1] = 1
                if flag == 0:
                    protos = proto
                    flag = 1
                else:
                    protos = np.vstack((protos,proto))
        clus = np.delete(clus, np.where(to_move)[0]).tolist()
        cols = np.delete(cols, np.where(to_move)[0]).tolist()
        # calculation of the Principal Activations
        pa = np.sum(protos,axis=0)
        nr = len(to_move)-sum(to_move)
        pa[pa<nr] = 0
        pa[pa==nr] = 1
        pa = np.reshape(pa,(1,-1))
        
        clusters.append(clus)
        non_significant.append(ns)
        prototypes.append(protos)
        principal_activations.append(pa)
        colors.append(cols)
    
    cimap_pas = {
           "name": cimap_out["name"],
            "clusters": clusters,
            "non_significant":non_significant,
            "prototypes": prototypes,
            "PA":principal_activations,
            "Colors":colors
       }
    return cimap_pas
    
def actplot(s):
    """
    Method for plotting all the cycles and the activation before the application of CIMAP
    
    :Input: * **s** (*dict*): the dictionary containing the cycles activation data created by the Input function from the *.csv* file.
    """
    # check that the field "Cycles" exists and is corectly labeled 
    keys = list(s.keys())
    keys_l = [k.lower() for k in keys]
    if not("cycles" in keys_l):
        raise ValueError('"Cycles" key of dictionary not found, check the input dictionary')
    elif not("Cycles" in keys) and "cycles" in keys_l:
        s["Cycles"] = s.pop("cycles")
        warnings.warn('"Cycles" field format wrong, corrected')
        
    # check that the field "Labels" exists and is corectly labeled 
    if not("labels" in keys_l):
        raise ValueError('"Labels" key of dictionary not found, check the input dictionary')
    elif not("Labels" in keys) and "labels" in keys_l:
        s["Labels"] = s.pop("labels")
        warnings.warn(' "Labels" field format wrong, corrected')
        
    for i,cyc in enumerate(s["Cycles"]):
        # setting the gaps to NaN so that it won't be graphed in any way and leaves
        # an empty space
        plt.figure()
        cycles = cyc.copy()
        cycles[cycles == 0] = np.nan
        X = np.linspace(0,100,cycles.shape[1])
        for j,cyc in enumerate(cycles):
            plt.plot(X,cyc*-j,'b', linewidth=1.5)
        plt.ylim((-j-1,1))
        plt.xlim((0,100))
        plt.title(s["Labels"][i])
        plt.yticks(ticks = np.arange(-j,1), labels = np.arange(j+1,0,-1))
        plt.ylabel('Gait cycles')
        plt.xlabel('% gait cycle')
        matplotlib.rc('ytick', labelsize=6)

def modality_distribution(s):
    
    """
    Method for the graphical representation of the distribution of the cycles into the modalities.

    :Input: * **s** (*dict*): the dictionary containing the cycles activation data created by the Input function from the *.csv* file.
    """
    # check that the field "Cycles" exists and is corectly labeled 
    keys = list(s.keys())
    keys_l = [k.lower() for k in keys]
    if not("cycles" in keys_l):
        raise ValueError('"Cycles" key of dictionary not found, check the input dictionary')
    elif not("Cycles" in keys) and "cycles" in keys_l:
        s["Cycles"] = s.pop("cycles")
        warnings.warn('"Cycles" field format wrong, corrected')
        
    # check that the field "Labels" exists and is corectly labeled 
    if not("labels" in keys_l):
        raise ValueError('"Labels" key of dictionary not found, check the input dictionary')
    elif not("Labels" in keys) and "labels" in keys_l:
        s["Labels"] = s.pop("labels")
        warnings.warn(' "Labels" field format wrong, corrected')
        
    for i,cycles in enumerate(s["Cycles"]):
        plt.figure()
        _,nact,_ = Intervals(cycles)
        plt.hist(nact, bins = np.arange(min(nact)-.5,max(nact)+1.5,1), rwidth = .5)
        plt.title(s["Labels"][i])
        plt.xlim((0,max(nact)+1))
        plt.xticks(ticks = np.arange(max(nact)+1))
        plt.ylabel('# of cycles')
        plt.xlabel('modalities')


def dendroplot(muscles):
    """
    Method for plotting the dendrograms built with CIMAP and chosen after performing the clustering process.

    :Input: * **muscles** (*dict*): the dictionary obtained as output from the CIMAP_cuts function.
    """
    # check for "dendrograms" and "clusters" field so that the input is correct
    keys = list(muscles.keys())
    if "dendrograms" not in keys:
        raise ValueError('"dendrograms" key of muscles not found, check "muscles" dictionary'
                         'be sure to run CIMAP_dendrograms and CIMAP_cuts first')
    if "clusters" not in keys:
        raise ValueError('"clusters" key of muscles not found, check "muscles" dictionary'
                         'be sure to run CIMAP_cuts first')
    
    
    for i,dens in enumerate(muscles['dendrograms']):
        count = 0
        for j,dendro in enumerate(dens):  
            if len(dendro):
                count +=1
        fig,axes = plt.subplots(count,1,figsize=(10,20))
        count = 0
        for j,dendro in enumerate(dens):
            if len(dendro):
                
                # retrieving the number of clusters
                ncl = muscles['clusters'][i][j].max()
                # calculating the threshold for coloring the clusters
                t = dendro[-1-(ncl-1),2]
                # setting the color palette
                link_color_pal = sns.color_palette("tab10")
                link_color_pal = list(link_color_pal.as_hex())
                link_color_pal.pop(3)
                link_color_pal.pop(1)
                hierarchy.set_link_color_palette(link_color_pal)
                # building the dendrogram graphically
                hierarchy.dendrogram(dendro,ax = axes[count],color_threshold = t+0.05, above_threshold_color='k')
                tit = str(j)+ ' modalities'
                axes[count].set_title(tit)
                count +=1
        fig.supxlabel('elements',fontsize=20)
        fig.supylabel('distance',fontsize=20)
        fig.suptitle(muscles['name'][i],fontsize=20)
        fig.tight_layout(pad = 1.5)
    


def clustersplot(cimap_out,color = False):
    """
    Method for the visualization of the clustering results of CIMAP.

    :Input: * **cimap_out** (*dict*): the dictionary obtained as output from the CIMAP_output function.
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
    
    # one muscle at a time
    for i,mslname in enumerate(cimap_out["name"]):
        # one modality at a time
        cont = 0
        for j,clus in enumerate(cimap_out["clusters"][i]):
            if bool(clus):
                cont+=1
        if cimap_out["non_significant"][i][0].any():
            cont+=1
        fig,axes = plt.subplots(1,cont,figsize=(20,10))
        cont = 0
        for j,clus in enumerate(cimap_out["clusters"][i]):
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
                        out,_,_ = Intervals(el)
                        proto = np.zeros((clus[0].shape[1]))
                        ins = np.median(out,axis = 0)*proto.shape[0]/100
                        ins -=1
                        el[el==0] = np.nan
                        # plotting the cluster
                        for jj,act in enumerate(el):
                            counter -=1
                            axes[cont].plot(X,act*counter, str(color_in[0]))
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
                        label.append('P' + str(k)+' - ' + str(idx[0]))
                        counter -=1
                axes[cont].set_yticks(tick) 
                axes[cont].set_yticklabels(label)
                axes[cont].set_ylim((counter-2,2))
                axes[cont].set_xlim((0,100))
                axes[cont].set_title(str(j)+" mod")
                cont +=1

        if cimap_out["non_significant"][i][0].any():
            # plotting the non_significant modalities with the reference to the
            # number of modality each cycles has

            ns = cimap_out["non_significant"][i][0].copy()
            idx = cimap_out["non_significant"][i][1]
            counter = 0
            tick = []
            label = []
            _,nact,_ = Intervals(ns)
             #ns[ns==0] = np.nan
            for ii,ac in enumerate(ns):
                 counter-=1
                 ac[ac==0] = np.nan
                 axes[cont].plot(X,ac*counter,'b')
                 tick.append(counter)
                 label.append(str(int(nact[ii]))+' mod - '+ str(int(idx[ii])))
            axes[cont].set_yticks(tick) 
            axes[cont].set_yticklabels(label)
            axes[cont].set_ylim((counter-2,2))
            axes[cont].set_xlim((0,100))
            axes[cont].set_title(" low number modalities")
        fig.suptitle(mslname,fontsize=20)
        fig.supxlabel("% gait cycle",fontsize=20)
        fig.supylabel("Gait cycles",fontsize=20)
        fig.tight_layout()

def PAsActivationsPlot(cimap_pas,color = False):
    """
    Method for the visualization of the clustering results of CIMAP and the Principal Activations extraction results.
    
    :Input: * **cimap_pas** (*dict*): the dictionary obtained as output from the PAs_extraction function.
            * **"color"** (*boolean*): False (*default*), parameter to set the color of the clusters matching their color on the dendrograms. Set to true if want the color to be changed.
    """

    # checking that the input is correct
    keys = list(cimap_pas.keys())
    if "name" not in keys:
        raise ValueError('"name" key of "cimap_pas" not found, check "cimap_pas" dictionary'
                         'be sure to run PAs_extraction first')
    if "clusters" not in keys:
        raise ValueError('"clusters" key of "cimap_pas" not found, check "cimap_pas" dictionary'
                         'be sure to run PAs_extraction first')
    if "non_significant" not in keys:
        raise ValueError('"non_significant" key of "cimap_pas" not found, check "cimap_pas" dictionary'
                         'be sure to run PAs_extraction first')
    if "prototypes" not in keys:
        raise ValueError('"prototypes" key of "cimap_pas" not found, check "cimap_pas" dictionary'
                         'be sure to run PAs_extraction first')
    if "PA" not in keys:
        raise ValueError('"PA" key of "cimap_pas" not found, check "cimap_pas" dictionary'
                         'be sure to run PAs_extraction first')
    
    for i,msl in enumerate(cimap_pas['name']):
        fig = plt.figure()
        # creating the X-axis reference
        X = np.linspace(0,100,cimap_pas['PA'][i].shape[1])
        counter = 0
        acts = cimap_pas['non_significant'][i].copy()
        acts[acts==0] = np.nan
        flag = True
        for ac in acts:
            # plotting the non significiant and getting the reference for the label
            if flag:
                plt.plot(X,ac*counter, color = [0.3, 0.3, 0.3],label = 'Non significant')
                flag = False
            else:
                plt.plot(X,ac*counter, color = [0.3, 0.3, 0.3])
            counter -=2
        counter -=2
        protos = cimap_pas['prototypes'][i].copy()
        protos[protos == 0] = np.nan
        flag = True
        # plotting the clusters
        for j,clus in enumerate(cimap_pas['clusters'][i]):
            clus = clus.copy()
            for acts in clus:
                acts[acts==0] = np.nan
                if color:
                    color_in = cimap_pas['Colors'][i][j]
                else:
                    color_in = 'b'
                plt.plot(X,acts*counter, color_in)
                counter -=2
            if flag:
                # plotting the prototype and getting the reference for the label
                plt.plot(X,protos[j,:]*counter,color = '#F48F0E', linewidth = 2, label = 'Prototype')
                flag = False
            else:
                plt.plot(X,protos[j,:]*counter,color = '#F48F0E', linewidth = 2)
            counter -=3
        counter -=3
        # plotting the principal activations
        pa = cimap_pas['PA'][i][0].copy()
        pa[pa==0] = np.nan
        plt.plot(X,pa*counter,'r', linewidth = 4, label = 'Principal Activation')
        plt.title(msl)
        plt.ylim((counter-5,5))
        plt.xlim((0,100))
        plt.xlabel('% gait cycle')
        plt.ylabel('Gait cycles')
        plt.yticks(ticks = [])
        fig.set_size_inches(6.5, 6)
        
        plt.legend(loc='upper right', bbox_to_anchor=(0.95, 1.05))


def targetgraph(targetdict,Target = 'All'):
    ''' Utility method that allow the user to decide what muscle given as input output graphically.

    :Input: * **targetdict** (*dict*): any dictionary given as input or obtained as output from CIMAP functions from which the user want to plot a muscle.
    
    :Configuration Parameter: * **"Target"**: "All" (*default*), labels or portion of the label of the muscle that you want to plot. For example if LGS_R and LGS_L are available inserting the target "LGS" would plot both. Inserting the exact label give as output the graphs related to that label.

    :Returns: * **outputdict** (*dict*): dictionary containing the information necessary for the graphical function to plot the desired muscle.
    '''
    outputdict = {}
    if Target.lower() == "all":
        ouputdict = targetdict.copy()
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
        
        idx = [i for i,label in enumerate(targetdict[name]) if Target in label]
        for k in keys:
            if k.lower() == 'subject':
                outputdict[k] = targetdict[k]
            else:
                outputdict[k] = [targetdict[k][ind] for ind in idx]
    return outputdict