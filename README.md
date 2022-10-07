# CIMAP Algorithm Toolbox
 
 The CIMAP (Clustering for Identification of Muscle Activation Patterns) algorithm is a method, based on agglomerative hierarchical clustering, developed for muscle activation pattern analysis during cyclical movements.
 What the algorithm does is group together cycles with a similar pattern of activation into clusters to identify common behavior in muscular activation.
 This result can also be elaborated for the identification of the principal activations (defined as those muscle activations that are strictly necessary to perform a specific task).
  The toolbox presented here provides a Python implementation of the CIMAP Algorithm with all the methods that allows to perform pattern analysis on muscle activation profiles.
 
 * **Documentation:** ...

## Insights
The current version includes the following functions:
- preprocessing of data,
- 
 
 ## Prerequisites
 CIMAP is run within a Python environment and dependent on a variety of libraries, most of which are contained within the installation of Python.
 Besides those libraries CIMAP relies on:
 >matplotlib==3.5.3 <br />
 >numpy==1.23.2 <br />
 >scipy==1.9.1 <br /> 
 >seaborn==0.12.0 <br /> 
 
 All the necessary libraries will be installed during the installation procedure, that will be explained in the following section.
 
 ## Installation
 All the installation steps should be done from the bash shell or the command prompt.
 ### Using pip
 To install CIMAP, the easiest way is to use pip:
```pip install CIMAP```
 

 #### Testing
``CIMAP`` uses the ``unittest`` package for testing
