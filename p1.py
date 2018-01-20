from pyspark.sql import SparkSession
from operator import add
import argparse
import numpy as np
import json

BAY = "Bayes"
LOG = "Logistic"
RAF = "RandomForest"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Assignment 0",
        epilog = "CSCI 8360 Data Science Practicum: Spring 2018",
        add_help = "How to use",
        prog = "python assignment0.py -i <input_directory> -o <output_directory> -m <mode> -s <stopfile_directory>")

    #required arguments
    parser.add_argument("-i", "--input", required = True,
        help = "The path to find the input data.")
    parser.add_argument("-o", "--output, required = True,"
        help = "The directory in which to place the output.")
    
    #optional arguments
    parser.add_argument("-c", "--classifier", choices = ["Bayes","Logistic","RandomForest"], default = "Bayes",
        help = "The type of classifier to use: Naive Bayes, Logistic Regression, or Random Forest")
    parser.add_argument("-r", "--regularize", action = store_true, default = False,
        help = "A flag for regularizing the feature space.")
    parser.add_argument("-s", "--smooth", action = store_true, default = False,
        help = "A flag for using a Laplace smoother on the input features.")
    
    args = vars(parser.parse_args())

    DATA_PATH = args['input'] 
    STOP_FILE = "stop/stopwords.txt"
    OUT_FILE  = args['output']

    CLASSIFIER = args['classifier']
    REGULARIZE = args['regularize']
    SMOOTH     = args['smooth']

    #TODO : Implement the classifiers

    spark = SparkSession\
        .builder\
        .appName("Project0")\
        .getOrCreate()

    sc = spark.sparkContext

    if ( CLASSIFIER == BAY ):
        #TODO : Implement Bayesian Classifier
        print()
    elif ( CLASSIFIER == LOG ):
        #TODO : Implement Logistic Classifier
        print()
    elif ( CLASSIFIER == RAF ):
        #TODO : Implement Random Forest Classifier
        print()
