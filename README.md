# CIMAP: Clustering for Identification of Muscle Activation Patterns

ADD LOGO HERE

The accurate temporal analysis of muscle activation is of great interest in several research areas, spanning from neurorobotic systems to the assessment of altered locomotion patterns in orthopedic and neurological patients and the monitoring of their motor rehabilitation. ```CIMAP``` is a python algorithm based on agglomerative hierarchical clustering that aims at characterizing muscle activation patterns during cyclical movements by grouping strides with similar muscle activity. More specifically, ```CIMAP``` allows for widening our understanding of muscle activation patterns by applying hierarchical clustering techniques to muscle activation intervals (i.e., onset and offset time-instants of muscle activations). From muscle activation intervals to the graphical representation of the clustering results, the proposed algorithm offers a complete analysis framework for assessing muscle activation patterns that can be flexibly modified at need and applied to cyclical movements different from walking. ```CIMAP``` is addressed to scientists of any programming skill level working in different research areas such as biomedical engineering, robotics, sports, clinics, biomechanics, and neuroscience.

## Installation
The latest stable release of ```CIMAP``` is freely available on [GitHub](https://github.com/marcoghislieri/CIMAP). Documentation and representative examples are freely available in each version’s readme file. The latest stable release of ```CIMAP``` can be easily installed through the bash shell or command prompt with the following commands:

-	Download [Python]( https://www.python.org/downloads/) and install (please have Python >= ```3.11.0```)
-	Open the bash shell or the command prompt and install the algorithm with ```pip install CIMAP```

Done! ```CIMAP``` is now correctly installed on your computer and ready to be used.

## What the ```CIMAP``` algorithm does:
1.	Data preparation (i.e., to read input data and convert them into the format needed for the following steps)
2.	Data pre-processing (i.e., to split input data based on the number of muscle activations within each cycle)
3.	Agglomerative hierarchical clustering:
      -	Cutting point identification based on the intra-cluster variability
      -	Distance metric selection based on the lowest inter-cluster variability
4.	Clusters' representation (available also at points 3 and 4)
5.	Data saving (clustering results are saved in an easy-to-read and open-source format)

<img  src="https://github.com/marcoghislieri/CIMAP/blob/main/docs/source/_static/CIMAPworkflow.png"/>

## Documentation
Detailed information on data requirements and algorithm functions can be found on the [Documentation](https://marcoghislieri.github.io/CIMAP/index.html) page. Moreover, a complete ``CIMAP``application example is provided.

## References
If you use ``CIMAP`` algorithm in your work, please cite the following two articles:

- S. Rosati, V. Agostini, M. Knaflitz, and G. Balestra, “Muscle activation patterns during gait: A hierarchical clustering analysis,” Biomed. Signal Process. Control, vol. 31, pp. 463–469, 2017, doi: [10.1016/j.bspc.2016.09.017](https://doi.org/10.1016/j.bspc.2016.09.017).

- S. Rosati, C. Castagneri, V. Agostini, M. Knaflitz, and G. Balestra, “Muscle contractions in cyclic movements: Optimization of CIMAP algorithm,” 2017, doi: [10.1109/EMBC.2017.8036762](https://doi.org/10.1109/EMBC.2017.8036762).

## Acknoledgments
The authors are grateful to (in alphabetical order): [Prof. Valentina Agostini](https://www.det.polito.it/personale/scheda/(nominativo)/valentina.agostini), Dr. Cristina Castagneri, and [Prof. Marco Knaflitz](https://www.det.polito.it/it/personale/scheda/(nominativo)/marco.knaflitz) for their contributions to the development and validation of this approach.

## How to contribute to ``CIMAP``
Thank you for your interest in our algorithm and for taking the time to read this document. Please refer to the [Contributing]( https://marcoghislieri.github.io/CIMAP/contribution.html) section for a complete guide on how to contribute to this algorithm.

## Disclaimer
This algorithm is provided as-is, and there are no guarantees that it fits your purposes or that it is bug-free.
