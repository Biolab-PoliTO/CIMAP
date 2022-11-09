# CIMAP: Clustering for Identification of Muscle Activation Patterns

## Summary

The main part of the movements of everyday life embodies a cyclical nature. Some examples are walking, running, cycling, stair climbing, and swimming. Gait analysis is broadly used for the clinical assessment of patients. To perform a quantitative assessment of muscle functionality during gait analysis, it is possible to  investigate non-invasively muscular activity by acquiring sEMG (surface electromyography). From the sEMG is possible, using several methods [[1]](https://doi.org/10.1109/10.661154), [[2]](https://doi.org/10.1186/s12984-021-00945-w), to identify the intervals when the muscle can be considered active or not. But, muscular activity, even during cyclical movements, such as walking, shows high intra-subject variability [[3]](https://doi.org/10.1016/j.gaitpost.2010.06.024) both in the number of activation intervals within the same cycle (also called modality) and in their duration. The high variability in the activation patterns of the same muscle while performing a cyclical task reduce the reliability and complicate the interpretability of the obtained results. To address such issue, Statistical Gait Analysis (SGA) [4] has been recently proposed. The SGA consists of a "statistical" description of gait Spatio-temporal parameters and parameters derived from EMG signals. The purpose of this approach is to describe gait functionality in a condition like everyday walking. From the SGA methods, the CIMAP (Clustering for Identification of Muscle Activation Pattern) [[5]](https://doi.org/10.1016/j.bspc.2016.09.017),[[6]](https://doi.org/10.1109/EMBC.2017.8036762) algorithm was developed to perform pattern analysis on the activation profiles extracted from the sEMG recording of walking. The aim CIMAP algorithm is to overcome the  limitation in interpretability introduced by the residual variability that is still present after the application of the SGA methods. The clustering performed on the activation intervals allows the identification of few common muscle activation patterns among all the cycles to give the user easier to interpret results. 
CIMAP is a Python algorithm based on agglomerative hierarchical clustering that aims at characterizing muscle activation patterns during cyclical movements by grouping strides with similar muscle activity. From muscle activation intervals to the graphical representation of the clustering results, the proposed algorithm offers a complete analysis framework for assessing muscle activation patterns that can be flexibly modified at need and applied to cyclical movements different from walking. CIMAP is addressed to scientists of any programming skill level working in different research areas such as biomedical engineering, robotics, sports, clinics, biomechanics, and neuroscience.

 This document describes briefly the CIMAP Algorithm (Clustering for Identification of Muscle Activation Patterns) concepts and the usage of the toolbox implementing it.
 
 The toolbox presented here provides a Python implementation of the CIMAP Algorithm with all the methods that allows to perform pattern analysis on muscle activation profiles.

* **Documentation**: [CIMAP Documentation]()
* **Tested OS**: Windows
 
 ## Workflow
 
 The typical workflow when using the CIMAP algorithm consists of the following 6 steps:
 
1.	Data preparation (i.e., to read input data and convert them into the format needed for the following steps);
2.	Data pre-processing (i.e., to split input data based on the number of muscle activations within each cycle);
3.	Agglomerative Hierarchical Clustering:
      1. Creation of a hierarchical tree using two distance metrics (Manhattan and Chebyshev);
4.	Selection of the optimal number of clusters;
      1.	Identification of the optimal cutting point through the analysis of both the intra-cluster variability;
      2.	Identification of the distance metric that has the best inter-cluster variability;
5.	Cluster representation (available also at points 3 and 4, see Figure 1 for a representative example);
6.	Data saving (clustering results are saved in an easy-to-read and open-source format).


 <img width="800" src="https://github.com/marcoghislieri/CIMAP/blob/master/docs/source/_static/CIMAPworkflow.png">
 
 ## Prerequisites
 CIMAP is run within a Python environment and dependent on a variety of libraries, most of which are contained within the installation of Python.
 Besides those libraries CIMAP relies on:
 
 >matplotlib>=3.5.3 <br />
 >numpy>=1.23.2 <br />
 >scipy>=1.9.1 <br /> 
 >seaborn>=0.12.0 <br /> 
 
 All the necessary libraries will be installed during the installation procedure, that will be explained in the following section.
 
 ## Availability
 The latest stable release of CIMAP is freely available on GitHub. Documentation and representative examples are freely available in each version’s readme file. The latest stable release of CIMAP can be easily installed through the bash shell or command prompt with the following command: ```pip install CIMAP```

## [Usage Example]()

## [How to contribute]()

### Testing
``CIMAP`` uses the ``unittest`` package for testing

## Please cite these papers
If you mean to use ``CIMAP`` please cite the following two articles.

[[5]](https://doi.org/10.1016/j.bspc.2016.09.017) S. Rosati, V. Agostini, M. Knaflitz, and G. Balestra, “Muscle activation patterns during gait: A hierarchical clustering analysis,” Biomed. Signal Process. Control, vol. 31, pp. 463–469, 2017, doi: 10.1016/j.bspc.2016.09.017

[[6]](https://doi.org/10.1109/EMBC.2017.8036762) S. Rosati, C. Castagneri, V. Agostini, M. Knaflitz, and G. Balestra, “Muscle contractions in cyclic movements: Optimization of CIMAP algorithm,” 2017, doi: 10.1109/EMBC.2017.8036762

### References

[[1]](https://doi.org/10.1109/10.661154) P. Bonato, T. D'Alessio and M. Knaflitz, "A statistical method for the measurement of muscle activation intervals from surface myoelectric signal during gait". IEEE Transactions on Biomedical Engineering.

[[2]](https://doi.org/10.1186/s12984-021-00945-w) M. Ghislieri, G.L. Cerone, M. Knaflitz et al., "Long short-term memory (LSTM) recurrent neural network for muscle activity detection". J. NeuroEngineering Rehabil.

[[3]](https://doi.org/10.1016/j.gaitpost.2010.06.024) V. Agostini, A. Nascimbeni, A. Gaffuri, P. Imazio, M.G. Benedetti, M. Knaflitz, Normative EMG activation patterns of school-age children during gait, Gait & Posture, Volume 32, Issue 3, 2010, Pages 285-289, ISSN 0966-6362, https://doi.org/10.1016/j.gaitpost.2010.06.024.
 
[4] V. Agostini, M. Knaflitz, R. U. Acharya, F. Molinari, T. Tamura, D. S. Naidu, & J. S. Suri (2012). Statistical gait analysis. Distributed diagnosis and home healthcare (D2H2), 2, 99-121.


## [License]()

## Disclaimer
This framework is provided as-is, and there are no guarantees that it fits your purposes or that it is bug-free.

