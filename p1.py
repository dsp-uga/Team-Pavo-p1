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
    
    sc = spark.sparkContext.getOrCreate()
    
    if ( CLASSIFIER == BAY ):
         
        X     = sc.textFile(DATA_PATH)
        Xtest = sc.textFile(TEST_DATA_PATH)
        y     = sc.textFile(LABEL_PATH)
        ytest = sc.textFile(TEST_LABEL_PATH)

        '''
        First zip with Index and reverse the tuple to achieve document labeling, 
        then flat map the values to get each word from each document with its label, 
        then map the values to lower case, remove the tedious formated quotation marks
        and strip any leading or trailing punctuation (not including apostrophes). 
        The final result is (did, wid) tuples
        '''
        X = X.zipWithIndex()\
             .map(lambda x: swap(x))\
             .flatMapValues(lambda x : x.split())\
             .mapValues(lambda x: x.lower().replace("&quot;","").strip(string.punctuation.replace("'","")))\
             .filter(lambda x: len(x[1])>1)
    
        Xtest = Xtest.zipWithIndex()\
                     .map(lambda x : (x[1],x[0]))\
                     .flatMapValues(lambda x : x.split())\
                     .mapValues(lambda x: x.lower().replace("&quot;","").strip(string.punctuation.replace("'","")))\
                     .distinct()
        
        
        '''
        Again, zip wth index and reverse the tuple to achieve document labeling, 
        flat map the values by splitting at the ",", 
        filter out any non "CAT" labels.
        The final result is (did, lab) tuples
        '''
        y = y.zipWithIndex()\
             .map(lambda x: (x[1],x[0]))\
             .flatMapValues(lambda x: x.split(","))\
             .filter(lambda x: not x[1].find("CAT")==-1)
            
        ytest = ytest.zipWithIndex()\
                     .map(lambda x: (x[1],x[0]))\
                     .flatMapValues(lambda x: x.split(","))\
                     .filter(lambda x: not x[1].find("CAT")==-1)
            
        #get the corpus vocabulary, the size of the vocabul, and the number of documents
        V = X.values().distinct()
        B = X.values().distinct().count()
        N = X.keys().distinct().count()

        #get the numer of documents of each label
        Nc = y.map(lambda x: (x[1],1)).reduceByKey(add)

        
        #get the estimated prior probabilities
        priors = Nc.mapValues(lambda x: x/N)
        priors.collect()

        #XX gets the number of occurances of each word for each doc (not identifying non-occurences)
        #  X   .map(...)         => ((wid,did),1) pairs for each (did,wid) pair in X
        #  ''  .reduceByKey(add) => ((wid,did),count1), **NOTE : No 0s**
        #  ''  .map(...)       => (did,(wid,count1)
        #  ''  .join(y)        => (did,((wid,count1),lab)
        #  ''  .mapValues(...) => (did,((lab,wid),count1)
        #  ''  .values()       => ((lab,wid),count)
        #  ''  .reduceByKey()  => ((lab,wid),count) where COUNT is the number of occurences of wid in lab type docs
        #  ''  .mapValues(...) => ((lab,wid),count+1) because we need to add one for LaPlace Smoothing
        VV = X.map(lambda x : (swap(x),1))\
              .reduceByKey(add)\
              .map(lambda x: (x[0][1],(x[0][0],x[1])))\
              .join(y)\
              .mapValues(lambda x: ((x[1],x[0][0]),x[0][1]))\
              .map(lambda x: x[1])\
              .reduceByKey(add)\
              .mapValues(lambda x: x+1)

        #We need to find out which words aren't in which docs
        #   y  .values()        => (lab)
        #  ''  .distinct()      => (lab)     unique labels
        #  ''  .cartesian(V)    => (lab,wid) unique labels
        #  ''  .subtract(...)   => (lab,wid) meaning wid never appeared in a lab type doc
        #  ''  .map(...)        => ((lab,wid),1) a fake word for LaPlace smoothing
        Missing = y.values()\
                   .distinct()\
                   .cartesian(V)\
                   .subtract(VV.keys())\
                   .map(lambda x: (x,1))

        #Join the labels back into the 
        VV = VV.union(Missing)


        #CBL gives us the total number of words in each class of document
        #  VV  .map(...)       => (lab,count')
        #  ''  .reduceByKey()  => (lab,COUNT)
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
                .mapValues(lambda x: (x[0])/(x[1]+B))

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
        cross = Xtest.keys()\
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
                     .map(lambda x: (x[1][0][0],(x[0],x[1][0][1]+np.log(x[1][1]))))\
                     .reduceByKey(NBFun6)

        #counting our success...
        cross = cross.map(lambda x: (x[0],x[1][0]))\
                     .join(ytest)          

        print(cross.mapValues(lambda x: (1 if x[0]==x[1] else 0))\
             .values()\
             .reduce(add))

    elif ( CLASSIFIER == LOG ):
        #TODO : Implement Logistic Classifier
        print()
    elif ( CLASSIFIER == RAF ):
        #TODO : Implement Random Forest Classifier
        print()
        
