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

def run_algorithm(input_file = None, color = True):
    ''' Function for the application of CIMAP to a dataset. This function when used applies all the methods of the algorithm to the data in the *input_file*.

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

        input_file = tkinter.filedialog.askopenfilename(parent = root, title = "Select Input File")
        root.destroy()
    # Reading of the data
    s,muscles = data_reading(input_file)
    # Removal of short intervals and fullon/fulloff cycles
    s = removeaddints(s)
    #division in modalities
    muscles = modalitydivision(s,muscles)
    # construction of the dendrograms
    muscles = dendrograms(muscles)
    # cut of the dendrogram and choice of the best clustering
    muscles = cuts(muscles)
    # output of the CIMAP
    cimap_out = algorithm_output(s,muscles)
    # save of the output file
    _ = resultsaver(cimap_out,input_file)
    print("CIMAP Algorithm application successfull")
    print("Graphical output generation")
    actplot(s)
    modality_distribution(s)
    dendroplot(muscles)
    clustersplot(cimap_out, color = color)
    plt.show(block = False)
    print("CIMAP graphical data produced")
    return

def intervals(cycles):
     '''
     Function for the extraction of the percentage value related to the activation intervals starting and ending point. This function is used in the pre-processing of the data for the extraction of the information necessary for the subsequent clustering steps. Also, the function returns the number of activation intervals in the cycle and the row where the cycle is put inside the "cycles" matrix. The row is used to mantain the sequence information of the cycles. 

      :Input: * **cycles** (*numpyarray*): a numpy binary array whose rows represents the gait cycles and the columns represent the samples of the normalised cycle. It is important that the cycles are normalised all at the same value, in our case 1000 time samples.

      :Output: * **out** (*list*): a list containing numpy arrays which contain the percentage value of the starting and ending point of the activation intervals (e.g., out[n] = [ON1,OFF1,...,ONn, OFFm])
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

def data_reading(input_file):
    '''Function that takes the input of CIMAP and prepares the data structures for the application of CIMAP Algorithm.
    
    :Input: * **input_file** (*string*): a string containing the path of the *.csv* file that contains the input data for the application of CIMAP. Be sure that the input data respects the requirements set in the Data Requirements section of the documentation.
    
    :Output: * **s** (*dict*): data structure of CIMAP that is used for the application of the methods of the algorithm.
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
    print("Input dataset loaded successfully")
    return s,muscles 

def _csv2dict(input_file):
    ''' Ausiliary function that opens and reads the contents of the *.csv* file and rearranges it for the application of CIMAP '''
    labels,cycles = [],[]
    with open(input_file,'r') as file:
        txt = file.read()
        file.seek(0)
        if ';' in txt:
            csvreader = csv.reader(file,delimiter = ';')
        else:
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

def removeaddints(s):
    """
    Function for the identification and removal of full active, full non-active cycles as this cycles are identified as outliers. The function also removes the activation intervals and small gaps that are smaller than 3% of the cycles as it was demonstrated in  `[1]`_ that such length in the activation is ininfluent on the biomechanics of the task and so they represent noise.

    .. _[1]: https://doi.org/10.1109/10.661154
    
    :Input: * **s** (*dict*): the dictionary containing the cycles activation data created by the Input function from the *.csv* file.

    :Output: * **s** (*dict*): same as the dictionary in input modified with the removal of the small gaps and of the always on/off cycles.
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
        cycles,nact,_ = intervals(cyc)
        
        # removal of the always OFF cycles
        if np.count_nonzero(nact==0):
            print("Full off cycles removed for muscle " +s["Labels"][f] + ": %s" % np.count_nonzero(nact==0))
            cycles = np.delete(cycles,nact==0,axis = 0)
            nact = np.delete(nact,nact==0,axis = 0)
        
    
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

        if bool(fullon):
            cycles = np.delete(cycles,fullon,axis = 0)
            nact = np.delete(nact,fullon,axis = 0)

        if bool(fullon):
            print("Full on cycles removed for muscle " +s["Labels"][f] + ": %s" % len(fullon))
        # Recreation of the cycles as binary arrays to be stored in s["Cycles] again
        cyc_out = np.zeros((cycles.shape[0],cyc.shape[1]))
        for k,ins in enumerate(cycles):
            ins = ins*cyc_out.shape[1]/100
            ins -=1
            for p in range(len(ins)):
                if p%2 == 0:
                    cyc_out[k,int(ins[p]):int(ins[p+1])+1] = 1
                    
        s["Cycles"][f] = cyc_out
    print("Pre-processing successfully performed")
    return s


def modalitydivision(s,muscles):
    """Function for the division of the gait cycles in the different modalities before clustering. The function uses the function intervals to retrieve the activation intervals and then it divides them inside a list in muscles where the index identifies the number of activation intervals of the modality (0 always empty).
    
    :Input: * **s** (*dict*): the dictionary containing the cycles activation data created by the Input function from the *.csv* file.
            * **muscles** (*dict*): dictionary obtained as output of CIMAP_input.
    :Output: * **muscles** (*dict*): dictionary that contains the cycles divided by modality represented by the value of edges of the activation intervals translated as percentage of the cycle values."""
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
             inters,nact,idx = intervals(c)
             sd = np.zeros(len(nact))+side[i][k]

             if flag == 0:
                intr = list(inters)
                acts = np.vstack((idx,sd,nact)).T
                flag = 1
             else:
                intr = intr+list(inters)
                a_ins = np.vstack((idx,sd,nact)).T
                acts = np.vstack((acts,a_ins))

        mods = []
        # sorting of the modalities
        for n in range(int(max(acts[:,-1])+1)):
            if any(acts[:,-1]==n):
                ins = np.array([])
                flag = 0
                for k,_ in enumerate(acts):
                    if acts[k,-1]==n:
                        if flag == 0:
                            ins = intr[k].T
                            flag = 1
                        else:
                            ins = np.vstack((ins,intr[k]))
                if len(np.where(acts[:,-1]==n)[0]) == 1:
                    ins =np.append(ins,acts[acts[:,-1]==n,:-1])
                    ins = np.reshape(ins,(1,-1))
                else:
                    ins = np.hstack((ins,acts[acts[:,-1]==n,:-1]))
            else:
                ins = np.array([])
            mods.append(np.array(ins, dtype=np.float))
        modalities.append(mods)
    muscles["modalities"] = modalities
    print("Cycles successfully divided into modalities")
    return muscles

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
        
def findcuts(distance):
    '''Function for the automatic identification of the cutting point on the dendrograms `[1]`_.

    :Input: * **distance** (*numpyarray*): the distance array that is the third column of the linkage function output.

    :Output: * **cuts** (*list*): list containing the optimal cutting point for each type of cut used.'''
    dist_diff = np.diff(distance)
    mean = np.mean(dist_diff[round(0.5*len(dist_diff))-1:])
    std = np.std(dist_diff[round(0.5*len(dist_diff))-1:], ddof=1)
    
    x = _smooth(dist_diff,5)
    i_sm = len(x)-1
    while x[i_sm]> x[i_sm-1] and i_sm > 0.8*len(x):
        if x[i_sm]> x[i_sm-1]:
            i_sm-=1
            
    
    idx3 = i_sm+2
    if any(list(dist_diff > mean)):
        idx1 = list(dist_diff > mean).index(True)+1
    else:
        idx1 = len(dist_diff)+1
    if any(list(dist_diff > mean+std)):
        idx2 = list(dist_diff > mean+std).index(True)+1
    else:
        idx2 = len(dist_diff)+1
    
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

def resultsaver(cimap_out,input_file = None, saving = True):
    '''Function for saving the results of CIMAP in a *.csv* file.

    :Input: * **cimap_out** (*dict*): the dictionary containing the results of the application of CIMAP obtained from the function algorithm_output.
            * **input_file** (*string*): the path of the input file containing the data given to CIMAP. When set to *None* the function gives the opportunity to choose the folder where to save the data and input manually the name to give to the file.
            * **saving** (*bool*): a boolean variable that can be used to decide whether to save the results or not.
    :Output: * **rows** (*array*): array containing the results of the application of CIMAP.'''
    
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
            _,nact,_ = intervals(cimap_out["non_significant"][i][0])
            # creating the 6 digits code for the non significant cycles
            row += list(map(lambda x: "{:0>2}".format(int(x))+"0000",nact))
            position += cimap_out["non_significant"][i][1].tolist()
        # rearraning to the sequential order of the cycles given in input
        row= [x for _,x in sorted(zip(position,row))]
        row.insert(0,cimap_out["name"][i])
        rows.append(row)
    # getting the path of the input file to write the file where the input_file is    
    if saving:
        if not(input_file):
            root = tkinter.Tk()
            root.attributes("-topmost", True)

            root.withdraw()
            path = tkinter.filedialog.askdirectory(parent = root, title='Select Folder')
            root.destroy()
            name_results = input("Please Insert the name of the file containig the results: ")
            f = open(path+"\\"+name_results+".csv", 'w')
        else:
            ps = os.path.dirname(input_file)
            f = open(ps+"\\"+ os.path.splitext(os.path.basename(input_file))[0]+"_Output_CIMAP.csv", 'w')
        
        
        writer = csv.writer(f,lineterminator='\r')
        writer.writerows(rows)
        f.close()
        print("Results saved")

    return rows

def actplot(s,target = 'All'):
    """
    Function for plotting all the cycles and the activation before the application of CIMAP
    
    :Input: * **s** (*dict*): the dictionary containing the cycles activation data created by the Input function from the *.csv* file.
            * **target**: "All" (*default*), labels or portion of the label of the muscle that you want to plot. For example if LGS_R and LGS_L are available inserting the target "LGS" would plot both. Inserting the exact label give as output the graphs related to that label.
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
    
    toplot = _targetgraph(s,target)    
    for i,cyc in enumerate(toplot["Cycles"]):
        # setting the gaps to NaN so that it won't be graphed in any way and leaves
        # an empty space
        plt.figure(figsize=(6,6))
        cycles = cyc.copy()
        cycles[cycles == 0] = np.nan
        X = np.linspace(0,100,cycles.shape[1])
        for j,cyc in enumerate(cycles):
            plt.plot(X,cyc*-j,'b', linewidth=2)
        plt.ylim((-j-1,1))
        plt.xlim((0,100))
        plt.title(toplot["Labels"][i])
        plt.yticks(ticks = np.arange(-j,1), labels = np.arange(j+1,0,-1),fontsize=6)
        plt.grid(alpha = 0.6)
        plt.ylabel('Gait cycles')
        plt.xlabel('Cycle duration (%)')
        matplotlib.rc('ytick', labelsize=6)
        plt.tight_layout()

def modality_distribution(s,target = 'All'):
    
    """
    Method for the graphical representation of the distribution of the cycles into the modalities.

    :Input: * **s** (*dict*): the dictionary containing the cycles activation data created by the Input function from the *.csv* file.
            * **target**: "All" (*default*), labels or portion of the label of the muscle that you want to plot. For example if LGS_R and LGS_L are available inserting the target "LGS" would plot both. Inserting the exact label give as output the graphs related to that label.

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
    
    toplot = _targetgraph(s,target)    
    for i,cycles in enumerate(toplot["Cycles"]):
        plt.figure()
        _,nact,_ = intervals(cycles)
        plt.hist(nact, bins = np.arange(min(nact)-.5,max(nact)+1.5,1), rwidth = .5)
        plt.title(toplot["Labels"][i])
        plt.xlim((0,max(nact)+1))
        plt.xticks(ticks = np.arange(max(nact)+1))
        plt.ylabel('Number of occurrences (#)')
        plt.xlabel('Modalities')


def dendroplot(muscles,target = 'All'):
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
    
    toplot = _targetgraph(muscles,target)
    for i,dens in enumerate(toplot['dendrograms']):
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
                ncl = toplot['clusters'][i][j].max()
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
                
                if toplot['metrics'][i][j][0] == 1:
                    cut = 'CutA'
                elif toplot['metrics'][i][j][0] == 2:
                    cut = 'CutB'
                elif toplot['metrics'][i][j][0] == 3:
                    cut = 'CutB'
                if j == 1:
                    tit = str(j)+ ' modality - ' + cut + ' - metric: ' + toplot['metrics'][i][j][1] 
                else:
                    tit = str(j)+ ' modalities - ' + cut + ' - metric: ' + toplot['metrics'][i][j][1] 
                axes[count].set_title(tit)
                count +=1
                plt.yticks(fontsize = 8)
                plt.xticks(fontsize = 8)
        fig.supxlabel('Cycles',fontsize=15)
        fig.supylabel('Distance',fontsize=15)
        fig.suptitle(toplot['name'][i],fontsize=15,x = 0.1, y=.95)
        fig.tight_layout(pad = 0.5)
    


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
    
    toplot = _targetgraph(cimap_out,target)
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


def _targetgraph(targetdict,target):
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
    