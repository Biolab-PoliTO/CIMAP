# CIMAP: Clustering for Identification of Muscle Activation Patterns

<img  src="https://github.com/Biolab-PoliTO/CIMAP/blob/main/docs/source/_static/Logo.png" width="600"/>

The accurate temporal analysis of muscle activation is of great interest in several research areas, spanning from neurorobotic systems to the assessment of altered locomotion patterns in orthopedic and neurological patients and the monitoring of their motor rehabilitation. However, due to the high intra-cycle variability of the muscle activation patterns, specific algorithms are needed to help scientists to easily characterize and assess muscle activation patterns during cyclical movements. ```CIMAP``` is a python algorithm based on agglomerative hierarchical clustering that aims at characterizing muscle activation patterns during cyclical movements by grouping strides with similar muscle activity. More specifically, ```CIMAP``` allows for widening our understanding of muscle activation patterns by applying hierarchical clustering techniques to muscle activation intervals (i.e., onset and offset time-instants of muscle activations). From muscle activation intervals to the graphical representation of the clustering results, the proposed algorithm offers a complete analysis framework for assessing muscle activation patterns that can be applied to cyclical movements different from walking. The algorithm can be flexibly modified at need to comply with the necessities of the user. ```CIMAP``` is addressed to scientists of any programming skill level working in different research areas such as biomedical engineering, robotics, sports, clinics, biomechanics, and neuroscience.

## Installation
The latest stable release of ```CIMAP``` is freely available on [GitHub](https://github.com/Biolab-PoliTO/CIMAP). Documentation and representative examples are freely available in each version’s readme file. The latest stable release of ```CIMAP``` can be easily installed through the bash shell or command prompt with the following commands:

-	Download [Python]( https://www.python.org/downloads/) and install (please have Python >= ```3.11.0```)
-	Open the bash shell or the command prompt and install the algorithm with ```pip install CIMAP```

### Using the setup.py file
You can download ```CIMAP``` from its GitHub repository as a zip file. A ```setup.py``` file (setuptools) is included in the root folder of ```CIMAP```. Enter the package's
root folder and run: ```python setup.py install```.

Done! ```CIMAP``` is now correctly installed on your computer and ready to be used.

## What the ```CIMAP``` algorithm does:
1.	Data preparation (i.e., to read input data and convert them into the format needed for the following steps)
2.	Data pre-processing (i.e., to split input data based on the number of muscle activations within each cycle)
3.	Agglomerative hierarchical clustering:
      -	Cutting point identification based on the intra-cluster variability
      -	Distance metric selection based on the lowest inter-cluster variability
4.	Clusters' representation (available also at points 3 and 4)
5.	Data saving (clustering results are saved in an easy-to-read and open-source format)

<img  src="https://github.com/Biolab-PoliTO/CIMAP/blob/main/docs/source/_static/CIMAPworkflow.png"/>

## Documentation
Detailed information on data requirements and algorithm functions can be found on the [Documentation](https://biolab-polito.github.io/CIMAP/index.html) page. Moreover, a complete ``CIMAP``application example is provided.

## Functions description
A detailed description of all the functions developed for performing CIMAP is available in the [Documentation](https://biolab-polito.github.io/CIMAP/index.html) in the [Algorithm Functions](https://biolab-polito.github.io/CIMAP/CIMAP.html) section.

## Application Example
A comprehensive application example is provided in the [Documentation](https://biolab-polito.github.io/CIMAP/index.html) page in the [Applying CIMAP to gait analysis](https://biolab-polito.github.io/CIMAP/example.html) section. The dataset on which ```CIMAP``` is applied is the same that is provided in the example_data folder. Following the same steps showed in the example section of the [Documentation](https://biolab-polito.github.io/CIMAP/index.html) it is possible to apply ```CIMAP``` to other datasets.

## References
If you use ```CIMAP``` algorithm in your work, please cite the following article:

- G. Dotti, M. Ghislieri, C. Castagneri, V. Agostini, M. Knaflitz, G. Balestra, and S. Rosati, "An Open-Source Toolbox for Enhancing the Assessment of Muscle Activation Patterns during Cyclical Movements",  Physiol. Meas. 2024, 45, 105004, doi:[10.1088/1361-6579/ad814f](https://doi.org/10.1088/1361-6579/ad814f).


## How to contribute to ``CIMAP``
Thank you for your interest in our algorithm and for taking the time to read this document. Please refer to the [Contributing]( https://biolab-polito.github.io/CIMAP/contribution.html) section for a complete guide on how to contribute to this algorithm.

## Disclaimer
This algorithm is provided as-is, and there are no guarantees that it fits your purposes or that it is bug-free.
