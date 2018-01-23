from pyspark.sql import SparkSession
from operator import add
import argparse
import numpy as np
import json
import string

BAY = "Bayes"
LOG = "Logistic"
RAF = "RandomForest"

'''
A method that takes in a tuple and returns a more sensible tuple. 
Changes n==None to n==0 and reverses some orderings. 
@param tup  the input tuple
@note       input tuple should be (wid,(n,did)) tuples
@return     (did,(wid,n)) tuples
'''
def NBFun1(tup):
    n = tup[1][0]
    did = tup[1][1]
    wid = tup[0]
    if n == None : n = 0
    return (did,(wid,n))
'''
A method that takes ina  tuple and returns a re-ordered tuple.
@param tup  the input tuple
@note       input should be (did,((wid,n),label)) 
@return     a tuple like ((wid,lab),n)
'''
def NBFun2(tup):
    wid = tup[1][0][0]
    n   = tup[1][0][1]
    lab = tup[1][1]
    return ((wid,lab),n)

'''
A method to get the conditional probability estimate for the Naive Bayesian classifier. 
@param tup  the input tuple
@note       the tuple should be (lab,((wid,count1),count2)) where count1 is the total 
            number of times wid appears in documents of type lab and count2 is the total 
            number of words in documents of type lab
@return     a tuple like ((wid,lab),P_hat(wid|lab))
'''
def NBFun3(tup,B):
    lab    = tup[0]
    wid    = tup[1][0][0]
    count1 = tup[1][0][1]
    count2 = tup[1][1]
    P      = count1 / (count2 + B)
    return ((wid,lab),P)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Project 1",
        epilog = "CSCI 8360 Data Science Practicum: Spring 2018",
        add_help = "How to use",
        prog = "python p1.py -i <input_data_file> -l <input_labal_file> -o <output_directory> [optional args]")

    #required arguments
    parser.add_argument("-i", "--inputData", required = True,
        help = "The path to find the input training data.")
    parser.add_argument("-l", "--inputLabels", required = True,
        help = "The path to find the input training labels.")
    parser.add_argument("-o", "--output", required = True,
        help = "The directory in which to place the output.")
    
    #optional arguments
    parser.add_argument("-y", "--testData", 
        help = "The path to find the input testing data.")
    parser.add_argument("-z", "--testLabels", 
        help = "The path to find the input testing labels.")
    parser.add_argument("-c", "--classifier", choices = ["Bayes","Logistic","RandomForest"], default = "Bayes",
        help = "The type of classifier to use: Naive Bayes, Logistic Regression, or Random Forest")
    parser.add_argument("-r", "--regularize", action = 'store_true', default = False,
        help = "A flag for regularizing the feature space.")
    parser.add_argument("-s", "--smooth", action = 'store_true', default = False,
        help = "A flag for using a Laplace smoother on the input features.")
    parser.add_argument("-t", "--stop",
        help = "The directory in which to find the stopwords file (if using).")
    
    args = vars(parser.parse_args())

    DATA_PATH = args['inputData']
    LABEL_PATH = args['inputLabels']
    OUT_FILE  = args['output']

    TEST_DATA_PATH  = args['testData']
    TEST_LABEL_PATH = args['testLabels']
    CLASSIFIER = args['classifier']
    REGULARIZE = args['regularize']
    SMOOTH     = args['smooth']
    STOP_FILE  = args['stop']    


    spark = SparkSession\
        .builder\
        .appName("Project0")\
        .getOrCreate()

    sc = spark.sparkContext

    if ( CLASSIFIER == BAY ):
        #TODO : Implement Bayesian Classifier
        print(args)
        if STOP_FILE != None : print(STOP_FILE)
        else : print("No stop file")

        #start with a list of documents, one document one each line of the input file
        X     = sc.textFile(DATA_PATH)
        Xtest = sc.textFile(TEST_DATA_PATH)
        #one string of labels for each document, though each document may have several labels
        y = sc.textFile(LABEL_PATH)
        '''
        First zip with Index and reverse the tuple to achieve document labeling, 
        then flat map the values to get each word from each document with its label, 
        then map the values to lower case, remove the tedious formated quotation marks
        and strip any leading or trailing punctuation (not including apostrophes). 
        The final result is (did, wid) tuples
        '''
        X = X.zipWithIndex()\
             .map(lambda x : (x[1],x[0]))\
             .flatMapValues(lambda x : x.split())\
             .mapValues(lambda x: x.lower().replace("&quot;","").strip(string.punctuation.replace("'","")))

        Xtest = Xtest.zipWithIndex()\
             .map(lambda x : (x[1],x[0]))\
             .flatMapValues(lambda x : x.split())\
             .mapValues(lambda x: x.lower().replace("&quot;","").strip(string.punctuation.replace("'","")))
        '''
        Again, zip wth index and reverse the tuple to achieve document labeling, 
        flat map the values by splitting at the ",", 
        filter out any non "CAT" labels. 
        '''
        y = y.zipWithIndex()\
             .map(lambda x: (x[1],x[0]))\
             .flatMapValues(lambda x: x.split(","))\
             .filter(lambda x: not x[1].find("CAT")==-1)

        #get the corpus vocabulary
        V = X.values()\
             .distinct()
        B = len(V.collect())
        VBroadcast = sc.broadcast(V.collect())

        #get the number of documents
        N = len(X.keys()\
                 .distinct()\
                 .collect())
        
        #get the numer of documents of each label
        Nc = y.map(lambda x: (x[1],1)).reduceByKey(add)

        #get the estimated prior probabilities
        priors = Nc.mapValues(lambda x: x/N)

        '''
        Get the number of times each word appears in each document class,
        making sure to find out if a word DOESN'T appear in a certain class.
        X is (did,wid) tuples.
        XX becomes (wid,did) tuples.
        V is (wid) tuples.
        V.map(...) becomes (wid,1) tuples.
        ''.roj(XX) yields (wid,(appears,did)) tuples
            where appears == None if wid doesn't appear in did and 1 if it does
        ''.map(NBFun1) yields (did,(wid,appears)) tuples
            where appears == 1 FOR EACH OCCURARNCE of wid in did, 
            or if wid does not appear in did, appears = 0 FOR A SINGLE TUPLE 
        ''.join(y) yields (did,((wid,appears),label)) tuples
        appearances.map(NBFun2) yields ((wid,label),appears) tuples
        ''.reduceByKey(add) yields ((wid,label),n) tuples, 
            i.e. - the number of occurences of wid in docs of type label
        ''.mapValues(lambda x: x+1) adds one to the above n for Laplace Smoothing
        Final tuple is ((wid,label),n+1)
        '''
        XX = X.map(lambda x : (x[1],x[0]))
        appearances = V.map(lambda x: (x,1))\
                       .rightOuterJoin(XX)\
                       .map(NBFun1)\
                       .join(y)
        Tct = appearance.map(NBFun2)\
                        .reduceByKey(add)\
                        .mapValues(lambda x: x+1)
        
        '''
        Get the total number of words in each document class.
        Tct.map(lambda x: (x[0][1],x[1]-1)) yields (label,n) tuples.
        ''.reduceByKey(add) yields (label,sum) tuples
            where sum is the number of words in all documents of type label
        '''
        Tct_by_label = Tct.map(lambda x: (x[0][1],x[1]-1))\
                          .reduceByKey(add)

        '''
        Get the conditional probabilities.
        Tct.map(...) yields (lab,(wid,count1)) tuples
        ''.join(Tct_by_label) yields (lab,((wid,count1),count2)) tuples
        ''.map(NBFun3) computes the probabilities and returns ((wid,lab),P_hat) tuples
        '''
        
        Ptc = Tct.map(lambda x: (x[0][1],(x[0][0],x[1])))\
                 .join(Tct_by_label)\
                 .map(lambda x: NBFun3(x,B))

        
        Xtest = Xtest.filter(lambda x: x[1] not in VBroadcast.value())
        
    elif ( CLASSIFIER == LOG ):
        #TODO : Implement Logistic Classifier
        print()
    elif ( CLASSIFIER == RAF ):
        #TODO : Implement Random Forest Classifier
        print()
        
