import numpy as np
import warnings

warnings.simplefilter("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore", category=UserWarning)

__all__ = ['intervals']

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
     if np.logical_and(cycles != 0, cycles != 1).any():
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