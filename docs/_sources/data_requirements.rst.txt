Data Requirements
=================

This section describes how the muscle activation data must be prepared for the application of CIMAP Algorithm.

Data Description
~~~~~~~~~~~~~~~~

The CIMAP algorithm has been mainly used for muscle activation pattern analysis on gait data. The method is formulated in a way that does not preclude the application to any kind of cyclical movement.
For the algorithm to work the data given as input must be prepared in a specific way. The first requirement is that the EMG signal is tranformed into activation profiles. The activation profiles are binary arrays that are equal to **1** when the muscle is considered active and **0** when the muscle is considered non-active. Here are two example methods that can be used for the identification of the muscle activation intervals: one based on a statistical approach `[1]`_ and one based on a Deep Learning approach `[2]`_.

.. _[1]: https://doi.org/10.1109/10.661154
.. _[2]: https://doi.org/10.1186/s12984-021-00945-w

Then is required that the cycles are time normalized to a total of 1000 samples. This is required for avoiding the influence of the natural difference in time duration of the different cycles. Also, the process standardizes in precision when defining the starting and ending points of the activation profiles.

Data Format
~~~~~~~~~~~

The :py:meth:`CIMAP.Run` method allows the user to select the files where the data is stored and automatically runs all the methods of CIMAP. The input data for CIMAP algorithm must be saved in a *.csv* file structured in the following way.
Each muscle will represent a row of the file with a total of Nx1000+1 columns, where N represents the total number of cycles. In each row the first column will be the label of each muscle. The label of each muscle, for an optimal functioning of the algorithm must to be written in this way: 'ID of the muscle' + '_' + 'L' or 'R' depending on the side (e.g., 'LGS_L' for the Left Lateral Gastrocnemius). From the second column to the end each column will contain the value 0 or 1 representing the muscle activation.

**Example:**

.. code:: text

    DATASET:
    col. 1			col. 2			col. 3			col. 4			col. 5			col. 6			col. 7			col. 8			...
    TA_L			1			1			1			1			1			1			1
    LGS_L 			0			0			0			0			0			0			0
    TA_R			0			0			1			1			1			1			1
    LGS_R	 		0			0			0			0			0			0			0


Output Data
~~~~~~~~~~~
The :py:meth:`CIMAP.Run` will automatically save a *.csv* file containing the results of the application of CIMAP named ' *NameInput* _CIMAP_Results.csv'. in the same folder where the input was.
The results of CIMAP will be structured the similarly to the input. The file will have a number of row equal to the number of muscles given as input. Each row will have a total of N+1 columns, where N represents the total number of cycles. In each row the first column will be the label of each muscle. The other coulmns, that represent the sequence of cycles as given in input, will contain a 6-digit ID that stores information as explained in the image: the first and the second digits represent the modality of the cycles, the others represents the number of the cluster of belonging of the cycle. In case of a cycle that is considered as non significant the 4 last digits will be all zeros.

.. figure:: /_static/digitscode.png
  :width: 300
  :align: center
  
  
**Output Example:**

.. code:: text

    DATASET:
    col. 1			col. 2			col. 3			col. 4			col. 5			...
    TA_L			020002			020001			030001			030001
    LGS_L 			020003			010001			010001			020002
    TA_R			030001			030001			020001			020002
    LGS_R	 		020003			020003			020003			020003


References
**********
1. P. Bonato, T. D'Alessio and M. Knaflitz, "A statistical method for the measurement of muscle activation intervals from surface myoelectric signal during gait," in IEEE Transactions on Biomedical Engineering, vol. 45, no. 3, pp. 287-299, March 1998, doi: 10.1109/10.661154.
2. M. Ghislieri, G.L. Cerone, M. Knaflitz et al. Long short-term memory (LSTM) recurrent neural network for muscle activity detection. J NeuroEngineering Rehabil 18, 153 (2021). https://doi.org/10.1186/s12984-021-00945-w