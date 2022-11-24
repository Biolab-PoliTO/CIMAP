Applying CIMAP to gait analysis
===============================

 Here is an example of how to use the CIMAP algorithm, no data are needed from you since the code uses a dataset freely available in the `GitHub <https://github.com/marcoghislieri/CIMAP/tree/main/example_data>`_ repository. More specifically, the CIMAP algorithm will be applied to study muscle activation patterns acquired from a healthy subject during overground walking at self-selected speed.

Importing libraries
^^^^^^^^^^^^^^^^^^^
 
  Be sure you have installed the latest stable release of the algorithm using the command ```pip install CIMAP``` command. through the bash shell or command prompt. You can check the release’s information using the command ```pip show CIMAP```. Then, you can import the library and set the input file path as follows:
 
 .. code-block:: python
 
	 # Import of the library
	 from CIMAP import CIMAP
	 
	 # Import of the library for the visualization of the graphical elements
	 import matplotlib.pyplot as plt
	 
	 input_file = ".\\example_data\\Dataset.csv"

 Notice that is possible to easily run the CIMAP algorithm without directly calling all of the CIMAP functions by using the function  :func:`~CIMAP.CIMAP.run_algorithm`.

Dataset preparation and pre-processing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 First, the :func:`~CIMAP.CIMAP.data_reading` function is used to open and format the input data according to the :doc:`Documentation <../index>` section. Within the :func:`~CIMAP.CIMAP.data_reading` function, the :func:`~CIMAP.CIMAP.intervals` function is used to extract muscle activation onset and offset (i.e., the beginning and the end of each muscle activation, respectively) time instants. Then, the :func:`~CIMAP.CIMAP.removeaddints` function is called to remove outliers (i.e., cycles characterized by always ON or always OFF activation patterns). Further details on the outliers removal process can be found in the study by Rosati *et al.* `[1]`_.

 .. _[1]: https://doi.org/10.1109/EMBC.2017.8036762
 
 .. code-block:: python
	
	 >>> s,muscles = CIMAP.data_reading(input_file = input_file)
	 Input dataset loaded successfully
	 
	 >>> s = CIMAP.removeaddints(s)
	 Pre-processing successfully performed

 Pre-processed muscle activation intervals can be graphically investigated using the :func:`~CIMAP.CIMAP.actplot` and :func:`~CIMAP.CIMAP.modality_distribution` functions. The :func:`~CIMAP.CIMAP.actplot` function represents the pre-processed muscle activation intervals over time for each acquired muscle and side, separately. If you are interested in a specific muscle, the target property of the :func:`~CIMAP.CIMAP.actplot` function can be used to set the muscle to be represented. As an example, only the muscle activation intervals extracted from the left and right Lateral Gastrocnemius (LGS) muscle are represented.
 

 
 .. code-block:: python
 
 	 # Plot muscle activation intervals
	 >>> CIMAP.actplot(s,target = 'LGS')
	 
	 # Command to display all open figures
	 >>> plt.show(block = False)
	 
 .. figure:: ./_static/Actplot.png
  :width: 800
  :align: center
   
  Muscle activation intervals extracted from the left and Lateral Gastrocnemius (LGS) muscle of a healthy subject during overground walking at self-selected speed. Blue lines represent muscle activation intervals normalized into 1000 time points with respect to the cycle duration. This representation was generated using ```CIMAP``` v1.0.4.

Cycles are then divided into several sub-datasets grouping together cycles showing the same number of muscle activations within the cycle (called *modalities*).

  .. code-block:: python
 
	 # Division of the cycles by modality
	 >>> muscles = CIMAP.modalitydivision(s,muscles)
	 Cycles successfully divided into modalities


The :func:`~CIMAP.CIMAP.modality_distribution` function, instead, can be used to represent the muscle activation patterns distributions. If you are interested in a specific muscle, the target property of the :func:`~CIMAP.CIMAP.modality_distribution` function can be used to set the muscle to be represented. As an example, only the histogram of the muscle activation patterns extracted from the left and right Lateral Gastrocnemius (LGS) muscle are represented.
 
 .. code-block:: python
	
 	 # Plot muscle activation patterns distributions
	 >>> CIMAP.modality_distribution(s,target = 'LGS')
	 
	 # Command to display all open figures
	 >>> plt.show(block = False)


 .. figure:: ./_static/Histograms.png
  :width: 800
  :align: center
  
  Occurrences of sEMG activation patterns of the left and right Lateral Gastrocnemius (LGS) muscle of a healthy subject during overground walking at self-selected speed. For each side, it is shown the number of gait cycles belonging to each modality. This representation was generated using ```CIMAP``` v1.0.4.

Agglomerative Hierarchical Clustering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

 Agglomerative hierarchical clustering is applied to each sub-dataset, separately. Using the :func:`~CIMAP.CIMAP.dendrograms` function, two different dendrograms are computed: the first one using the Manhattan distance metric and the second one using the Chebychev distance metric. Then, the cutoff point for each of the two dendrograms and the best clustering results are chosen using the :func:`~CIMAP.CIMAP.cuts` function. Further details on the identification of the cutoff point and the selection of the best clustering results can be found in the :doc:`Documentation <../index>` section.

 
  .. code-block:: python
 
	 # Building dendrograms
	 >>> muscles = CIMAP.dendrograms(muscles)
	 Dendrograms building completed
	 
	 # Choice of the best clustering results
	 >>> muscles = CIMAP.cuts(muscles)
	 Best clustering result chosen


 Clustering results can be graphically represented through the :func:`~CIMAP.CIMAP.dendroplot` and :func:`~CIMAP.CIMAP.clustersplot functions. The :func:`~CIMAP.CIMAP.clustersplot` function plots the hierarchical tree of each computed modality. Clusters obtained after the selection of the optimal cutoff point are represented in different colours. If you are interested in a specific muscle, the target property of the :func:`~CIMAP.CIMAP.dendroplot` function can be used to set the muscle to be represented. As an example, only the clustering results computed from the left and right Lateral Gastrocnemius (LGS) muscle are represented.
 

 
 .. code-block:: python
 
 	 # Dendrogram representation
	 >>> CIMAP.dendroplot(muscles,target = 'LGS')
	 
	 # Command to display all open figures
	 >>> plt.show(block = False)


 .. figure:: ./_static/Dendros.png
  :width: 800
  :align: center

  Dendrograms of hierarchical cluster analysis performed on cycles showing a single activation interval (top) and on cycles showing two different activation intervals (bottom), separately. Clusters obtained after the selection of the optimal cutoff point are represented in different colours. SEMG activation intervals were extracted from the Lateral Gastrocnemius (LGS) muscle of a representative healthy subject during a 5-minute overground walking at a self-selected speed. This representation was generated using ```CIMAP``` v1.0.4.


The :func:`~CIMAP.CIMAP.clustersplot` function, instead, can be used to show the original muscle activation intervals grouped in clusters and divided by modality. results of clustering representing the activation intervals group in each cluster divided by modality. The color property of the :func:`~CIMAP.CIMAP.clustersplot` function can be used to have a color map consistent with the one represented using the :func:`~CIMAP.CIMAP.dendroplot` function. If you are interested in a specific muscle, the target property of the :func:`~CIMAP.CIMAP.clustersplot` function can be used to set the muscle to be represented. As an example, only the clustering results computed from the left and right Lateral Gastrocnemius (LGS) muscle are represented.


 .. code-block:: Python
 
	 # Obtain the output of the algorithm
	 >>> cimap_out = CIMAP.algorithm_output(s,muscles)
	 Output dictionary created
	 
	 # Plot muscle activation intervals grouped in clusters and divided by modality
 	 >>> CIMAP.clustersplot(cimap_out,target = 'LGS', color = True)
	 
	 # Command to display all open figures
	 >>> plt.show(block = False)

 .. figure:: ./_static/Clusters.png
  :width: 800
  :align: center

  Representation of muscle activation intervals grouped in clusters and divided by modality. For each cluster, is represented the centroid (black lines) identified by the label ‘P’ + ‘N’, where N is the number associated to the cluster. The single-element clusters are represented as centroids, thicker but still coloured. The cycle belonging to the modalities that did not have enough cycles to build a dendrogram on are represented in the ‘Modality under Th = 10’ panel. This representation was generated using ```CIMAP``` v1.0.4.


Data saving
^^^^^^^^^^^

 Finally, to save the output data, the :func:`~CIMAP.CIMAP.resultsaver` function should be used. This function has the property *input_file* set equal to “*None*” by default. When called, it will open a window that allows you to select a folder where to save the results.


 .. code-block:: python
 
	 # Save clustering results
	 >>> CIMAP.resultsaver(cimap_out)
	 Please insert the name of the file containing the results: results_file
	 Results saved

 All the code presented in this tutorial will work as in the example if you copy and paste it into your Python IDE.

References
**********

`[1]`_. S. Rosati, C. Castagneri, V. Agostini, M. Knaflitz, and G. Balestra, “Muscle contractions in cyclic movements: Optimization of CIMAP algorithm,” 2017, doi: 10.1109/EMBC.2017.8036762.