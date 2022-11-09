from CIMAP import CIMAP
import unittest
import numpy as np
import csv
import warnings



class TestCIMAPInput(unittest.TestCase):
    def setUp(self):
        input_file = ".\\test_data\\Dataset.csv"
        self.s, self.muscles = CIMAP.data_reading(input_file=input_file)

    def test_shapes(self):
        self.assertEqual(np.shape(list(self.s.keys()))[0], 2)
        self.assertEqual(np.shape(self.s['Labels'])[0], 10)
        self.assertEqual(np.shape(self.s['Cycles'])[0], 10)
        self.assertEqual(self.s['Cycles'][0].shape[1], 1000)
        self.assertEqual(np.shape(list(self.muscles.keys()))[0], 3)
        self.assertEqual(np.shape(self.muscles['name'])[0], 5)


class TestCIMAPOutput(unittest.TestCase):

    def Test_output(self):
        out_file = ".\\test_data\\Dataset_Output_CIMAP.csv"
        rows = []
        with open(out_file, 'r') as file:
            csvreader = csv.reader(file)
            for row in csvreader:
                rows.append(row)
        file.close()
        input_file = ".\\test_data\\Dataset.csv"
        self.s, self.muscles = CIMAP.data_reading(input_file=input_file)
        self.s = CIMAP.removeaddints(self.s)
        #division in modalities
        self.muscles = CIMAP.modalitydivision(self.s, self.muscles)
        # construction of the dendrograms
        self.muscles = CIMAP.dendrograms(self.muscles)
        # cut of the dendrogram and choice of the best clustering
        self.muscles = CIMAP.cuts(self.muscles)
        # output of the CIMAP
        self.cimap_out = CIMAP.algorithm_output(self.s, self.muscles)
        # save of the output file
        outrows = CIMAP.resultsaver(cimap_out, input_file, saving=False)
        self.assertEqual(np.shape(outrows), np.shape(rows))
        for outrow in outrows:
            for row in rows:
                if outrow[0] == row[0]:
                    self.assertEqual(outrow, row)


if __name__ == "__main__":
	warnings.simplefilter("ignore", category=np.VisibleDeprecationWarning)
	unittest.main()
