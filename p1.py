from pyspark.sql import SparkSession
from operator import add
import argparse
import numpy as np
import json
import string

BAY = "Bayes"
LOG = "Logistic"
RAF = "RandomForest"

def swap(tup):
    return (tup[1],tup[0])

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

def NBFun4(tup):
    wid = tup[0]
    n = 1 if not tup[1][0] == None else 0
    did = tup[1][1]
    return ((wid,did),n)

def NBFun5(tup):
    count = tup[1]
    if count == None : count = 0
    return count

def NBFun6(accum,n):
    ret = accum if accum[1]>n[1] else n
    return ret

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
        #start with a list of labels, one string of labels for each document, though each document may have several labels
        y = sc.textFile(LABEL_PATH)
        ytest = sc.textFile(TEST_LABEL_PATH)
        '''
        First zip with Index and reverse the tuple to achieve document labeling, 
        then flat map the values to get each word from each document with its label, 
        then map the values to lower case, remove the tedious formated quotation marks
        and strip any leading or trailing punctuation (not including apostrophes). 
        The final result is (did, wid) tuples
        '''
        print(X.collect())
        X = X.zipWithIndex()\
             .map(lambda x: swap(x))\
             .flatMapValues(lambda x : x.split())\
             .mapValues(lambda x: x.lower().replace("&quot;","").strip(string.punctuation.replace("'","")))\
             .filter(lambda x: len(x[1])>1)
    
        Xtest = Xtest.zipWithIndex()\
                     .map(lambda x : swap(x))\
                     .flatMapValues(lambda x : x.split())\
                     .mapValues(lambda x: x.lower().replace("&quot;","").strip(string.punctuation.replace("'","")))\
                     .filter(lambda x: len(x[1])>1)\
                     .distinct()
        
        '''
        Again, zip wth index and reverse the tuple to achieve document labeling, 
        flat map the values by splitting at the ",", 
        filter out any non "CAT" labels.
        The final result is (did, lab) tuples
        '''
        y = y.zipWithIndex()\
             .map(lambda x: swap(x))\
             .flatMapValues(lambda x: x.split(","))\
             .filter(lambda x: not x[1].find("CAT")==-1)
        ytest = ytest.zipWithIndex()\
                     .map(lambda x: swap(x))\
                     .flatMapValues(lambda x: x.split(","))\
                     .filter(lambda x: not x[1].find("CAT")==-1)

        #get the corpus vocabulary, the size of the vocabul, and the number of documents
        V = X.values()\
             .distinct()
        B = V.count()
        N = X.keys()\
             .distinct()\
             .count()

        #get the numer of documents of each label
        Nc = y.map(lambda x: (x[1],1)).reduceByKey(add)

        #get the estimated prior probabilities
        priors = Nc.mapValues(lambda x: x/N)

        #XX gets the number of occurances of each word for each doc (not identifying non-occurences)
        #X     .map(...)         => ((wid,did),1) pairs for each (did,wid) pair in X
        #  ''  .reduceByKey(add) => ((wid,did),count1), count1 the number of times wid appears in did
        #                           NOTE: No count1 will equal 0
        XX = X.map(lambda x : (swap(x),1))\
              .reduceByKey(add)
        
        #VV helps us find the zero occurances
        #V     .cartesian(...) => (wid,did) **NOTE : each unique**
        #  ''  . map(...)      => ((wid,did),1) 
        #  ''  .loj(XX)        => ((wid,did),(1,count1)) if (wid,did) is a key in XX
        #                                               OR
        #                         ((wid,did),(1,None))   if (wid,did) is NOT a key in XX
        #  ''  .mapValues(...) => ((wid,did),count2)   
        #  ''  .map(...)       => (did,(wid,count2)) 
        #  ''  .join(y)        => (did,((wid,count2),lab)
        #  ''  .mapValues(...) => (did,((lab,wid),count2)
        #  ''  .values()       => ((lab,wid),count2)
        #  ''  .reduceByKey()  => ((lab,wid),COUNT) where COUNT is the number of occurences of wid in lab type docs
        
        VV = V.cartesian(X.keys().distinct())\
              .map(lambda x: (x,1))\
              .leftOuterJoin(XX)\
              .mapValues(lambda x: x[1] if not x[1]==None else 0)\
              .map(lambda x: (x[0][1],(x[0][0],x[1])))\
              .join(y)\
              .mapValues(lambda x: ((x[1],x[0][0]),x[0][1]))\
              .values()\
              .reduceByKey(add)

        #CBL gives us the total number of words in each class of document
        #            => (lab,COUNT2)
        countByLabel = VV.map(lambda x: (x[0][0],x[1]))\
                 .reduceByKey(add)

        #Tct is the conditional probabilities
        #VV.map(...)            => (lab,(wid,COUNT))
        #  ''  .join(CBL)       => (lab,((wid,COUNT),COUNT2))
        #  ''  .map(...)        => ((lab,wid),(COUNT1,COUNT2))
        #  ''  .mapValues(...)  => ((lab,wid),P(wid|lab))
        Tct = VV.map(lambda x: (x[0][0],(x[0][1],x[1])))\
                .join(countByLabel)\
                .map(lambda x: ((x[0],x[1][0][0]),(x[1][0][1],x[1][1])))\
                .mapValues(lambda x: (x[0]+1)/(x[1]+B))

        #This gets us the probability estimates P(lab = c | x) for each c
        #Xtest.keys() => (did)      NOT unique
        #  ''  .distinct() => (did)     UNIQUE
        #  ''  .cartesian(...) => (did,lab) for each did and lab value
        #  ''  .join(Xtest)    => (did,(lab,wid))
        #  ''  .map(swap)      => ((lab,wid),did)
        #  ''  .join(Tct)      => ((lab,wid),(did,P(wid|lab))
        #  ''  .map(...)       => ((lab,did),P(wid|lab))
        #  ''  .mapValues(...) => ((lab,did),log(P(wid|lab)))
        #  ''  .reduceByKey(...) => ((lab,did),Sum_wid(log(P(wid|lab))))
        #  ''  .map(...)         => (lab,(did,Sum_wid))
        #  ''  .join(priors)     => ((lab,((did,Sum_wid),P(lab))))
        #  ''  .map(...)         => (did,(lab,Sum_wid + P(lab)))
        #  ''  .reduceByKey(NBFun6) => (did,lab) where lab has Max Sum_wid + P(lab) over all lab for this did
        probs = Xtest.keys()\
                     .distinct()\
                     .cartesian(y.values().distinct())\
                     .join(Xtest)\
                     .map(swap)\
                     .join(Tct)\
                     .map(lambda x: ((x[0][0],x[1][0]),x[1][1]))\
                     .mapValues(lambda x: np.log(x))\
                     .reduceByKey(add)\
                     .map(lambda x: (x[0][0],(x[0][1],x[1])))\
                     .join(priors)\
                     .map(lambda x: (x[1][0][0],(x[0],x[1][0][1]+x[1][1])))\
                     .reduceByKey(NBFun6)
        
        
    elif ( CLASSIFIER == LOG ):
        #TODO : Implement Logistic Classifier
        print()
    elif ( CLASSIFIER == RAF ):
        #TODO : Implement Random Forest Classifier
        print()
        
