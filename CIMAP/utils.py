import csv
import numpy as np

__all__ = ['csv2dict','smooth']

def csv2dict(input_file):
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

def smooth(a,WSZ):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ    
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))
