.. CIMAP Algorithm documentation master file, created by
   sphinx-quickstart on Wed Sep 21 17:40:48 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

CIMAP Algorithm Toolbox
=======================
 The CIMAP (Clustering for Identification of Muscle Activation Patterns) algorithm is a method, based on agglomerative hierarchical clustering, developed for muscle activation pattern analysis during cyclical movements.
 What the algorithm does is group together cycles with a similar pattern of activation into clusters to identify common behavior in muscular activation.
 This result can also be elaborated for the identification of the principal activations (defined as those muscle activations that are strictly necessary to perform a specific task).
 
Algorithm Workflow
------------------

 CIMAP is an algorithm, based on Statistical Gait Analysis, that was developed and later optimized by Rosati et al. to perform pattern analysis on muscle activation profiles.
 The method address the problem of the high intra-subject variability that each subject has when performing a repetitive task.
 For the algorithm to work properly the cycles that are given as input to CIMAP has to be time-normalized to a fixed number of sample to remove the bias introduced
 by biological time differences in the performance of the task.
 
 The first step of the algorithm is the identification of the activation intervals edges to transform them into percentage values of the cycle. Then the cycles are divied by modality, i.e. the number of activation intervals whithin a cycle identify the modality (e.g., a cycles with n activation intervals is a n-modality cycle) to have higher homogeneity in the elaboration. After the division in modalities the clustering is performed on each modality separately, only on the modalities that present a sufficient number of cycles. The threshold Th on the minimum number of cycles per modality to perform clustering is set to Th = 10, as it was recently to showed to not influence significantly the results and allows the application to low number of cycles acquisitions (Dotti et al.). For each modality two dendrograms, one for each possible distance metric that are used, are built using the "complete" method (i.e., using the farthest distance between every pair of elements). Two distance metrics are used (Manhattan and Chebyshev) because during the optimization process was demostrated that neither perfomed significantly better than the other.
 Then, the number of clusters is chosen automatically based on the difference in distance of the merged clusters. Considering the difference of inter-cluster distance between consecutive iterations. Of the three cutoff points the best one is identified by finding the one that shows the best compromise between low intra-cluster variability and a high number of cycles included in the significant clusters (clusters with more than one element). After the choice of the best cutoff, the best clustering obtain from the two distance metrics is identified.
 The distance metrics is chosen as the one the shows the lowest mean similarity between the centroid and the cluster elements.
 Then, if selected, the results from clustering can be elaborated for the extraction of the principal activations. First, the significant clusters are identified as those clusters that contain more than 10% of the total number of cycles that were considered as significant in the previous elaboration. Subsequently the principal activations are extracted as the intersection of the activation intervals of the prototypes from the significant clusters.


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
   
   notebook/CIMAP_tutorial


References
==========
1. S. Rosati, V. Agostini, M. Knaflitz, and G. Balestra, “Muscle activation patterns during gait: A hierarchical clustering analysis,” Biomed. Signal Process. Control, vol. 31, pp. 463–469, 2017, doi: 10.1016/j.bspc.2016.09.017.
2. S. Rosati, C. Castagneri, V. Agostini, M. Knaflitz, and G. Balestra, “Muscle contractions in cyclic movements: Optimization of CIMAP algorithm,” 2017, doi: 10.1109/EMBC.2017.8036762.
3. V. Agostini, S. Rosati, C. Castagneri, G. Balestra, and M. Knaflitz, “Clustering analysis of EMG cyclic patterns: A validation study across multiple locomotion pathologies,” in 2017 IEEE International Instrumentation and Measurement Technology Conference (I2MTC), May 2017, pp. 1–5, doi: 10.1109/I2MTC.2017.7969746.
4. G. Dotti, M. Ghislieri, S. Rosati, V. Agostini, M. Knaflitz, G. Balestra, (2021, November). The Effect of Number of Gait Cycles on Principal Activation Extraction. In Annual International Conference of the IEEE Engineering in Medicine and Biology Society. IEEE Engineering in Medicine and Biology Society. Annual International Conference (Vol. 2021, pp. 985-988)..
