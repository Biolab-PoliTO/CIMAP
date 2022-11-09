.. CIMAP Algorithm documentation master file, created by
   sphinx-quickstart on Wed Sep 21 17:40:48 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

CIMAP: Clustering for Identification of Muscle Activation Patterns
==================================================================
 The CIMAP (Clustering for Identification of Muscle Activation Patterns) algorithm is a method, based on agglomerative hierarchical clustering, developed for muscle activation pattern analysis during cyclical movements.
 What the algorithm does is group together cycles with a similar pattern of activation into clusters to identify common behavior in muscular activation.
 
Algorithm Workflow
------------------

 CIMAP is an algorithm, that was developed `[1]`_ and optimized `[2]`_ to perform pattern analysis on muscle activation intervals during gait analysis. The method addresses the issue of the high intra-subject variability that each subject has when performing a cyclical task `[3]`_, as it is while walking. The current version of the algorithm can be flexibly modified at need and applied to cyclical movements different from walking. 
 For the algorithm to work properly the cycles that are given as input to CIMAP has to be time-normalized to a fixed number of sample to remove the bias introduced by biological time differences in the performance of the task. All the information regarding the data format are given in the :doc:`Data Requirements <../data_requirements>` section. In case of bilateral acquisitions the cycles belonging to the same muscle, but from different sides, are merged and all the processing steps are applied as they were from the same muscle. The cycles are divided again into sides when obtaining the results of the application of the algorithm. 
 
 .. _[1]: https://doi.org/10.1016/j.bspc.2016.09.017
 .. _[2]: https://doi.org/10.1109/EMBC.2017.8036762
 .. _[3]: https://doi.org/10.1016/j.gaitpost.2010.06.024
 
Data pre-processing
^^^^^^^^^^^^^^^^^^^
 
 The first step of the algorithm is the identification of the activation intervals edges to transform them into percentage values of the cycle. A process of cleaning of the activation intervals is performed by removing the gaps between activation intervals and those activation intervals within each cycle that are shorter than 3% of the gait cycles. Also, cycles that show a constant behaviour over the whole duration of the cycle (always active or always non-active) are removed. Then the cycles are divied by modalities, according to the number of activation intervals within a cycle (e.g., a cycles with *n* activation intervals is a *n-modalities* cycle) to have higher homogeneity in the pattern matching processes performed later in the algorithm.
 
Agglomerative Hierarchical Clustering 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 
 After the division in modalities the clustering is performed on each modality separately, only on the modalities that present a sufficient number of cycles. The threshold (Th) on the minimum number of cycles per modality to perform clustering is set to Th = 10, as it was recently to showed that lowering the number of cycles taken into account do not influence significantly the results and allows the application to situations where the patient is able to perform only fewer numbers of cycles during acquisitions `[4]`_). For each modality two dendrograms, one for each possible distance metric that are used, are built using the "complete" method. Two distance metrics are used (Manhattan and Chebyshev).
 
  .. _[4]: https://doi.org/10.1109/EMBC46164.2021.9629818
 

   
Selection of the optimal number of clusters  
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 
 The final number of clusters is chosen automatically based on the difference in distance of the merged clusters considering the difference of inter-cluster distance between consecutive iterations `[2]`_. For each distance metrics, three cutoff points are identified and the best one is chosen by finding the one that shows the best compromise between low intra-cluster variability and a high number of cycles excluding the clusters with one element. After the choice of the best cutoff, the best clustering obtained from the two distance metrics is identified. The distance metrics is chosen as the one that shows the lowest mean similarity between the centroid and the cluster's elements.

 

.. figure:: ./_static/CIMAPworkflow.png
  :width: 800
  :align: center
  
   
.. toctree:: :titlesonly:
   :maxdepth: 2
   :caption: Documentation:
   
   data_requirements
   CIMAP
   
.. toctree:: :titlesonly:
   :maxdepth: 4
   :caption: Examples:
   
   example

.. toctree:: :titlesonly:
   :maxdepth: 4
   :caption: About:
   
   contribution
   license





PLEASE CITE THESE PAPERS:
=========================

`[1]`_. S. Rosati, V. Agostini, M. Knaflitz, and G. Balestra, “Muscle activation patterns during gait: A hierarchical clustering analysis,” Biomed. Signal Process. Control, vol. 31, pp. 463–469, 2017, doi: 10.1016/j.bspc.2016.09.017.

`[2]`_. S. Rosati, C. Castagneri, V. Agostini, M. Knaflitz, and G. Balestra, “Muscle contractions in cyclic movements: Optimization of CIMAP algorithm,” 2017, doi: 10.1109/EMBC.2017.8036762.

Additional Papers
-----------------

`[3]`_. V. Agostini, A. Nascimbeni, A. Gaffuri, P. Imazio, M.G. Benedetti, M. Knaflitz, Normative EMG activation patterns of school-age children during gait, Gait & Posture, Volume 32, Issue 3, 2010, Pages 285-289, ISSN 0966-6362, https://doi.org/10.1016/j.gaitpost.2010.06.024.

`[4]`_. G. Dotti, M. Ghislieri, S. Rosati, V. Agostini, M. Knaflitz, G. Balestra, (2021, November). The Effect of Number of Gait Cycles on Principal Activation Extraction. In Annual International Conference of the IEEE Engineering in Medicine and Biology Society. IEEE Engineering in Medicine and Biology Society. Annual International Conference (Vol. 2021, pp. 985-988)..
