Data Requirements
=================

This section describes how your muscle activation data must be pre-processed in order to be used as input of CIMAP algorithm.
To work properly, the algorithm requires that the data given as input must be prepared in a specific way. The first requirement is that the sEMG signal is tranformed into activation intervals. The activation intervals are binary arrays that are equal to **1** when the muscle is considered active and **0** when the muscle is considered non-active. Here are two representative methods that can be used for the identification of the muscle activation intervals: one based on a statistical approach `[1]`_ and one based on a Deep Learning approach `[2]`_.
Then is required that the cycles are time normalized to a total of 1000 time-samples. This is required for avoiding the influence of the natural difference in time duration of the different cycles.

.. _[1]: https://doi.org/10.1109/10.661154
.. _[2]: https://doi.org/10.1186/s12984-021-00945-w


Data Format
~~~~~~~~~~~

Input Data
^^^^^^^^^^

The :py:meth:`CIMAP.Run` method allows the user to select the file where the data is stored and automatically runs all the methods of CIMAP. The input data for CIMAP algorithm must be saved in a *.csv* file structured in the following way. The file should contain as many rows as the number of muscles observed and as many columns as the number of time-samples acquired plus one, because the first column contains the label referring to the muscle name.
Each muscle will represent a row of the file with a total of Nx1000+1 columns, where N represents the total number of cycles. In each row the first column will be the label of each muscle. The label of each muscle should be formatted as it follows: [*Name of the muscle*, "_", "L" or "R"] depending on the side (e.g., 'LGS_L' for the Left side Lateral Gastrocnemius). Muscles with the same *Name of the muscle* are the ones that have the cycles merged for the application of CIMAP. The side "L" or "R" is used as information for the division when obtaining the results of the algorithm application.


**Input Example:**

Following there is an example extracted from the example dataset that is available in the `GitHub`_ repository.

.. _GitHub: https://github.com/marcoghislieri/CIMAP

+---------------------------------------------------------------------------------------------------+
|DATASET                                                                                            |
+========================+=========+=========+=========+=========+=========+=========+=========+====+
|column 1                |column 2 |column 3 |column 4 |column 5 |column 6 |column 7 |column 8 |... |
+------------------------+---------+---------+---------+---------+---------+---------+---------+----+
|TA_L                    |1        |1        |1        |1        |1        |1        |1        |... |
+------------------------+---------+---------+---------+---------+---------+---------+---------+----+
|LGS_L                   |0        |0        |0        |0        |0        |0        |0        |... |
+------------------------+---------+---------+---------+---------+---------+---------+---------+----+
|TA_R                    |1        |1        |1        |1        |1        |1        |1        |... |
+------------------------+---------+---------+---------+---------+---------+---------+---------+----+
|LGS_R                   |0        |0        |0        |0        |0        |0        |0        |... |
+------------------------+---------+---------+---------+---------+---------+---------+---------+----+


Output Data
^^^^^^^^^^^
The :py:meth:`CIMAP.Run` will automatically save a *.csv* file containing the results of the application of CIMAP named ' *NameInput* _CIMAP_Results.csv'. in the same folder where the input file was.
The results of CIMAP will be structured similarly to the input file. The output file will have a number of row equal to the number of muscles given as input. Each row will have a total of N+1 columns, where N represents the total number of cycles. In each row, the first column will be the label of each muscle. The other coulmns, that represent the sequence of cycles as given in input, will contain a 6-digit ID that stores information as explained in the image: the first and the second digits represent the modality of the cycles, the others represents the number of the cluster of belonging of the cycle. In case of a cycle that is considered as non significant the 4 last digits will be all zeros. The figure shows an example of how the output data is formatted.


.. figure:: /_static/digitscode.png
  :width: 300
  :align: center
  
  
**Output Example:**

Following is an example extracted from the results of the application of the algorithm on the example dataset that is available in the `GitHub`_ repository.
   
+---------------------------------------------------------------------+
|DATASET                                                              |
+========================+=========+=========+=========+=========+====+
|column 1                |column 2 |column 3 |column 4 |column 5 |... |
+------------------------+---------+---------+---------+---------+----+
|TA_L                    |030005   |040001   |030001   |030010   |... |
+------------------------+---------+---------+---------+---------+----+
|LGS_L                   |010012   |020002   |010011   |010014   |... |
+------------------------+---------+---------+---------+---------+----+
|TA_R                    |040001   |020002   |030002   |030009   |... |
+------------------------+---------+---------+---------+---------+----+
|LGS_R                   |020001   |010012   |010001   |010016   |... |
+------------------------+---------+---------+---------+---------+----+
   


References
**********
`[1]`_. P. Bonato, T. D'Alessio and M. Knaflitz, "A statistical method for the measurement of muscle activation intervals from surface myoelectric signal during gait". IEEE Transactions on Biomedical Engineering.

`[2]`_. M. Ghislieri, G.L. Cerone, M. Knaflitz et al., "Long short-term memory (LSTM) recurrent neural network for muscle activity detection". J. NeuroEngineering Rehabil.