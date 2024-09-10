from .data_reading import data_reading
from .find_cuts import find_cuts
from .cuts import cuts
from .remove_add_ints import remove_add_ints
from .modality_division import modality_division
from .dendrograms import dendrograms
from .algorithm_output import algorithm_output
from .result_saver import result_saver
from .act_plot import act_plot
from .modality_distribution import modality_distribution
from .dendro_plot import dendro_plot
from .clusters_plot import clusters_plot

import matplotlib
import matplotlib.pyplot as plt
import warnings, csv,os, tkinter, tkinter.filedialog

__all__ = ['run_algorithm']

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
    s = remove_add_ints(s)
    #division in modalities
    muscles = modality_division(s,muscles)
    # construction of the dendrograms
    muscles = dendrograms(muscles)
    # cut of the dendrogram and choice of the best clustering
    muscles = cuts(muscles)
    # output of the CIMAP
    cimap_out = algorithm_output(s,muscles)
    # save of the output file
    _ = result_saver(cimap_out,input_file)
    print("CIMAP Algorithm application successfull")
    print("Graphical output generation")
    act_plot(s)
    modality_distribution(s)
    dendro_plot(muscles)
    clusters_plot(cimap_out, color = color)
    plt.show(block = False)
    print("CIMAP graphical data produced")
    return