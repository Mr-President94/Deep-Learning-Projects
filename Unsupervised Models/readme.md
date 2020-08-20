# Unsupervised Deep Learning Projects
This folder contains Unsupervised Deep Learning Projects in a simplified format

# 1. Credit Card Fraud Detection using Self-Organzied Map

### Objective :
* Suppose we are Deep Learning Scientists working at a bank and we are given a datatset regarding th customers of this bank who are applying for an advanced Credit Card
* This data would be info that a customer provies while applying for the card
* And our job is to detect potential frauds among these applications
* In the end we should be able to give a list to the Manager of the customers who cheated on their application

### Goals of this project  
* Unlike the Machine Learning Models, where we predict whether each customer might be a potential fradulent by already training our machine based on previously labelled fraudster as YES or NO, here our approach would be to not consider the dependent variable and plot a map which shows us what a fraudster would look like
* We will use an **Unsupervised Deep Learning Algorithm** called **Self-Organzing Map**, which means we will be identifying patterns in high dimensional datasets full of non linear relationships and one of these patterns(in this case the customers) will be the potential frauds or the customers who cheated
* In more simple terms I will be doing Customer Segmentation to identify segments of customers and one of the segments will contain the customers who cheated

**Some other goals inculde -** 
* Understanding the idealogy of Self Organizing Map
* Implementing concepts like - Winning Node, Dimensionality Reduction, Mean Inter-neuron Distance 
* Interpreting outliers in the SOM 
* Understanding the need of Normalization
* Interpreting the SOM in detail and making it more interactive by mapping the info from the dataset
* Inverse mapping, inverse scaling and getting the list of potential fraudsters

# 2. Movie Recommender System using Boltzmann Machines

### Objective
* The objective here is to create a Recommender System with a binary output YES/NO to predict whether or not the viewer will like a particular movie
* This project is sort of a Deep Case Study on RBM which focuses purely upon implementing the theoritical knowledge of a Restricted Boltzmann Machine into Python

### Data Source
* GroupLens Research has collected and made available rating data sets from the MovieLens web site (http://movielens.org). The data sets were collected over various periods of time, depending on the size of the set. 
* It has 1 million ratings from 6000 users on 4000 movies. Released in FEB/2003. 

### Why Pytorch?
* Is has proven more useful and effective than tensorflow when it comes to implementing Boltzmann Machines and Auto Encoders
* Theano and Tensorflow can be used to implement the same but the framework of Pytorch makes implementation more easier
* It is more inutuitive, practical and flexible to make any changes in the architecture of your model when it comes to Pytorch

### Goals of this Project :
* Implementing and knowing the significance of Pytorch and performing Tensor operations
* To make our own class to implement Restricted Boltzmann Machines
* Understanding how an RBM works 
* Understanding the techniques of k-step contrastive divergence, updating the weights, probabilities of hidden and visible nodes, etc
* Calculating the loss function and interpreting the loops and what happens at the backend of an RBM
