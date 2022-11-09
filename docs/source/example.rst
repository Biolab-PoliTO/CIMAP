CIMAP Application Example
=========================

 In this tutorial we are going to make an example of application of CIMAP algorithm to some data.
 The Data is stored in the folder \example_data under the name "Dataset.csv". The data is an example of the data that can be used with CIMAP. The dataset given stores the activation intervals of N gait cycles acquired over 5 muscles  bilaterally on a healthy subject during a walk. The 5 muscles are: Tibialis Anterior (TA), Gastrocnemius Lateralis (LGS), Rectus Femoris (RF), Lateral Hamstring (LH) and Gluteus Medius (GMD). The activation intervals have been identified using a deep learning approach `[1]`_.

 .. _[1]: https://doi.org/10.1186/s12984-021-00945-w
 
  First be sure you have installed the CIMAP using ```pip install CIMAP``` command.
  The following command retrieves the informations about the installed package.
  
  .. code-block:: python
  
	 >>> pip show CIMAP
	Name: CIMAP
	Version: 0.1.3
	Summary: A Python package for muscle activation pattern analysis
	Home-page: 
	Author: Gregorio Dotti
	Author-email: gregorio.dotti@polito.it
	License: MIT
	Location: c:\users\d047490\appdata\local\programs\python\python310\lib\site-packages
	Requires: matplotlib, numpy, scipy, seaborn
	Required-by:
	

 Before the application of the algorithm import the library and set the input file path.
 
 .. code-block:: python
 
	 # import of the library
	 from CIMAP import CIMAP
	 # import of the library for the visualization of the graphical elements
	 import matplotlib.pyplot as plt
	 # To simply run the algorithm without making all the steps is possible to use the function CIMAP.run_algorithm(). This function allows the user to choose the input data file and run all the methods at once.
	 input_file = ".\\example_data\\Dataset.csv"


Data pre-processing
^^^^^^^^^^^^^^^^^^^

 The first step of the algorithm is using the :func:`~CIMAP.CIMAP.data_reading` function to open the data, formatted according to the :doc:`Data Requirements <../data_requirements>` section. Within the :func:`~CIMAP.CIMAP.data_reading` function, the :func:`~CIMAP.CIMAP.intervals` function is used to tranform the activation intervals edges into percentage values of the cycle. The second function that is applied is the :func:`~CIMAP.CIMAP.removeaddints` function. It performs cleaning on the data by removing two categories of elements. The first element removed are the gaps between activation intervals and those activation intervals within each cycle that are shorter than 3% of the gait cycles. This value was defined in `[2]`_ as biomechanically non relevant activation in gait. The second element removed are those cycle that present a constant behavior over the whole duration of the cycle (i.e., the cycle is always active or there is no activation over the cycle).

 .. _[2]: https://doi.org/10.1109/10.661154

 .. code-block:: python
	
	 >>> s,muscles = CIMAP.data_reading(input_file = input_file)
	 Input dataset loaded successfully
	 >>> s = CIMAP.removeaddints(s)
	 Interval and outliers removal performed

 The loaded data informations can be investigated using the graphical functions :func:`~CIMAP.CIMAP.actplot` and :func:`~CIMAP.CIMAP.modality_distribution`. The :func:`~CIMAP.CIMAP.actplot` function shows the cycles of the input data sequentially highlighting where the muscle is considered active and where not. The :func:`~CIMAP.CIMAP.actplot` function parameter *target* can be used to set which of the muscle to represent specifically. In our case, the shown muscle is only the LGS for summarizing convenience.
 
 .. code-block:: python
 
 	 >>> CIMAP.actplot(s,target = 'LGS')
	 >>> plt.show(block = False)
	 
 .. figure:: ./_static/Actplot.png
  :width: 800
  :align: center
   
  Representation of the activation intervals computerd from the Gastrocnemius Lateralis of a healthy subject during a walk. The blue lines represent where the muscle was identified as active.

The :func:`~CIMAP.CIMAP.modality_distribution` function shows the histogram of the cycles of the input data divided into modalities. The :func:`~CIMAP.CIMAP.modality_distribution` function parameter *target* can be used to set which of the muscle to represent specifically. In our case, the shown muscle is only the LGS for summarizing convenience.
 
 .. code-block:: python
 
 	 >>> CIMAP.modality_distribution(s,target = 'LGS')
	 >>> plt.show(block = False)


 .. figure:: ./_static/Histograms.png
  :width: 800
  :align: center
  
  Histograms representing the distribution of the cycles among the modalities.
  
Then the cycles are divied by modalities to have higher homogeneity in the pattern matching processes performed later in the algorithm. At this step the same muscle, but from different sides in case of bilateral aacquisition, are merged.

 .. code-block:: python
 
	 # Division of the cycles by modality
	 >>> muscles = CIMAP.modalitydivision(s,muscles)
	 Cycles successfully divided into modalities

Agglomerative Hierarchical Clustering and Selection of the optimal number of clusters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 After the division in modalities the clustering is performed on each modality separately only on the modalities that present a sufficient number of cycles (threshold Th = 10). The clustering is performed first by building the two dendrograms using the :func:`~CIMAP.CIMAP.dendrograms` function, one for each distance measure. Then the best clustering is chosen using the :func:`~CIMAP.CIMAP.cuts` function. The clustering is chosen as the one that shows the best compromise between low intra-cluster variability and a high number of cycles excluding those that end in single element clusters.
 
  .. code-block:: python
 
	 # Building the hierarchical tree
	 >>> muscles = CIMAP.dendrograms(muscles)
	 Dendrograms building completed
	 # Choice of the best clustering results
	 >>> muscles = CIMAP.cuts(muscles)
	 Best clustering result chosen

 For the visualization of the results of the clustering the graphical functions  :func:`~CIMAP.CIMAP.dendroplot` and :func:`~CIMAP.CIMAP.clustersplot` can be used. The :func:`~CIMAP.CIMAP.dendroplot` plots the hierarchical tree of each modality on which it was built and the clusters are coloured to represent the clustering result. The cutting point can be identified as the highest non-black part of the hierarchical tree. The :func:`~CIMAP.CIMAP.dendroplot` function parameter *target* can be used to set which of the muscle to represent specifically. In our case, the shown muscle is only the LGS for summarizing convenience.
 
 .. code-block:: python
 
 	 >>> CIMAP.dendroplot(muscles,target = 'LGS')
	 >>> plt.show(block = False)


 .. figure:: ./_static/Dendros.png
  :width: 800
  :align: center

  Dendrograms reporting the clustering results in colorcode. For each dendrogram is shown which cut and norm resulted in the best clustering `[3]`_.
 
  .. _[3]: https://doi.org/10.1109/EMBC.2017.8036762

The :func:`~CIMAP.CIMAP.clustersplot` function shows the results of clustering representing the activation intervals group in each cluster divided by modality. In case of modalities on which the dendrogram was not built the cycles are all represented aside as "low number modalities". The :func:`~CIMAP.CIMAP.clustersplot` function parameter *color* can be set to *True* to have the clusters cloured matching the dendrograms colors. The :func:`~CIMAP.CIMAP.clustersplot` function parameter *target* can be used to set which of the muscle to represent specifically. In our case, the shown muscle is only the LGS for summarizing convenience.

 .. code-block:: python
 
	 # obtain the output of the algorithm
	 >>> cimap_out = CIMAP.algorithm_output(s,muscles)
	 Output dictionary created, use resultsaver to save the data in .csv format
 	 >>> CIMAP.clustersplot(cimap_out,target = 'LGS', color = True)
	 >>> plt.show(block = False)

 .. figure:: ./_static/Clusters.png
  :width: 800
  :align: center

  Representation of the activation intervals divided into clusters. For each cluster is represented the centroid (thicker in black) identified by the label 'P' + '*N*', where *N* is the number associated to the cluster. The single element clusters are represented as centroids, thicker but still coloured. The cycle belonging to the modalities that did not have enough cycles to build a dendrogram on are represented in the 'low number modalities' labeled with the associated number of modalities.
  
In the end, to save the data the :func:`~CIMAP.CIMAP.resultsaver` function can be used. The function has the parameter *input_file* set equal to *None* by default. If given the path of the input file as value to the *input_file* the function automatically saves the results  in the same position of the input_file. Otherwise, the function opens a window that allows to select a folder where to save the results and is asked the user to insert a name for the file.

 .. code-block:: python
 
	 
	 >>> CIMAP.resultsaver(cimap_out)
	 Please Insert the name of the file containig the results: results_file
	 Results saved


References
**********
`[1]`_. P. Bonato, T. D'Alessio and M. Knaflitz, "A statistical method for the measurement of muscle activation intervals from surface myoelectric signal during gait". IEEE Transactions on Biomedical Engineering.

`[2]`_. M. Ghislieri, G.L. Cerone, M. Knaflitz et al., "Long short-term memory (LSTM) recurrent neural network for muscle activity detection". J. NeuroEngineering Rehabil.

`[3]`_. S. Rosati, C. Castagneri, V. Agostini, M. Knaflitz, and G. Balestra, “Muscle contractions in cyclic movements: Optimization of CIMAP algorithm,” 2017, doi: 10.1109/EMBC.2017.8036762.