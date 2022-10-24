---
title: 'CIMAP: A Python package for muscle activation pattern analysis'
tags:
  - Python
  - Muscle activation patterns
  - Surface EMG
  - Hierarchical Clustering
authors:
  - name: 
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: "1" #


affiliations:
 - name: Biolab Laboratory, Department of Electronics and Telecommunication, Politecnico di Torino, Italy
   index: 1
date: 
bibliography: paper.bib

---

# Summary

The main part of the movements of everyday life embodies a cyclical nature. Some examples are walking, running, cycling, stair climbing, and swimming. In particular, gait analysis is broadly used for the clinical assessment of patients. The acquisition of sEMG (surface Electromyography) during walking is used for assessing muscular functionality. From the sEMG is possible, using spcifically developed methods {}{}, to identify the intervals when the muscle can be considered active or not. But, muscular activity, even during the performance of cyclical movements, such as walking, shows high intra-subject variability {} both in the number of activation intervals and in their length.  Recently Statistical Gait Analysis (SGA) {} has been recently proposed. The SGA consists of a "statistical" description of gait spatio-temporal parameters and parameters derived from EMG signals. The purpose of this apporoach is to describe gait funcnionality in a condition similar to everiday walking. From the SGA methods CIMAP (Clustering for Identification of Muscle Activation Pattern) algorithm was developed to perform pattern analysis on the activation profiles extracted from sEMG recording of walking.
CIMAP is an algorithm based on agglomerative hierarchical clustering that groups together gait cycles that show similar onset-offset intervals to form clusters. First the algorithm  Each cluster identifies a particular pattern and characterizes the muscular behavior during the cyclical movement. Even though the algorithm was developed for gait analysis, its application is not limited to that field, and it can be applied to all sorts of cyclical movements.


# Statement of Need