import csv,os, tkinter, tkinter.filedialog
from .intervals import intervals 

__all__ = ['result_saver']

def result_saver(cimap_output,input_file = None, saving = True):
    '''Function for saving the results of CIMAP in a *.csv* file.

    :Input: * **cimap_output** (*dict*): the dictionary containing the results of the application of CIMAP obtained from the function algorithm_output.
            * **input_file** (*string*): the path of the input file containing the data given to CIMAP. When set to *None* the function gives the opportunity to choose the folder where to save the data and input manually the name to give to the file.
            * **saving** (*bool*): a boolean variable that can be used to decide whether to save the results or not.
    :Output: * **rows** (*array*): array containing the results of the application of CIMAP.'''
    
    rows= []
    # for each muscle
    for i in range(len(cimap_output["name"])):
        
        row,position = [],[]
        # for each modality
        for j,cl in enumerate(cimap_output["clusters"][i]):
            if bool(cl):
                # creating the 6 digits code for each cycle
                row += list(map(lambda x: "{:0>2}".format(j)+"{:0>4}".format(x),cl[1]))
                position+=  cl[3].tolist()
        # checking the non significant cycles        
        if cimap_output["non_significant"][i][0].any():
            _,nact,_ = intervals(cimap_output["non_significant"][i][0])
            # creating the 6 digits code for the non significant cycles
            row += list(map(lambda x: "{:0>2}".format(int(x))+"0000",nact))
            position += cimap_output["non_significant"][i][1].tolist()
        # rearraning to the sequential order of the cycles given in input
        row= [x for _,x in sorted(zip(position,row))]
        row.insert(0,cimap_output["name"][i])
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