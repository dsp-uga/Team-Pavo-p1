Scalable Document Classification with Naive Bayes in Spark (UGA-CSCI-8360-Project1)
===================================================================================
In project 1, we are using the Reuters Corpus, which is a set of news stories split into a hierarchy of categories. There are multiple class labels per document, our goal is to build a naive bayes model without using any build-in pacakges such as MLLib or scikit-learn. The model achieves 94.7% prediction accuracy in the hold-out data set.

The data
===========
There are multiple class labels per document, but for the sake of simplicity we’ll ignore all but the labels ending in CAT: CCAT, GCAT, MCAT and ECAT. There are some documents with more than one CAT label. Treat those documents as if you observed the same document once for each CAT label (that is, add to the counters for all the observed CAT labels). Here are the available data sets:  

X_train_vsmall.txt, y_train_vsmall.txt  
X_test_vsmall.txt, y_test_vsmall.txt  
X_train_small.txt, y_train_small.txt  
X_test_small.txt, y_test_small.txt  
X_train_large.txt, y_train_large.txt  
X_test_large.txt  


Procedure
===========
There are three parts in this project: data cleaning/pre-processing, naive bayes modeling, and prediction.

###Data Cleaning/Pre-processing   
We take the following steps to clearn and process the raw data,
Remove all special characteristics, punctuations, and stopping words.

###Naive Bayes Modeling  
We applied Multinomial Naive Baye method, by assuming the independence of p(xi|cj), where xi is the word in a document d, and the cj is the jth category. The idea is that MAP Category = argmax P(cj)*Π(P(xi|cj); where P(cj) is:  (docu. number in Cj)/(total docu. number); Π(P(xi|cj) = p(x1|cj)*p(x2|cj)*...*p(xn|cj), where xi is the word in this document and Cj is the word's category.  

The word counts are calculated with Spark RDD operations.  
Build Laplace smoothing (A.K.A add 1 method) naive bayes model with word counts.  
Take log () of the probabilities to overcome the underflow problem for large data set.  

###Prediction  
For large test data, take the same data cleaning procedures as training set.  
Predict with the trained model and write the results in a single text file.  
Submit the text file with predictions to Autolab and get the accuracy results.  


Using Google CLoud
===========
Data Storage- where main python file is saved along with sample data

DataProc- Creating a cluster and submitting a job

Before creating a cluster make sure a billing account is added to that project.
Open Google Cloud consloe--Billing--add billing details.

Create a cluster-gcloud dataproc clusters create cluster-name
Manually set master and worker configuration by using GCP console.

Setting up a Job:
gcloud dataproc jobs submit spark --cluster cluster-name -mainpthonfile.py-arguments
 

Authorship
===========
Team Pavo:  
    Hiten Nirmal  
    Nicholas Klepp  
    Jin Wang  
