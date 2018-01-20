from pyspark.sql import SparkSession
from operator import add
import argparse
import numpy as np
import json

BAY = "Bayes"
LOG = "Logistic"
RAF = "RandomForest"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Project 1",
        epilog = "CSCI 8360 Data Science Practicum: Spring 2018",
        add_help = "How to use",
        prog = "python assignment0.py -i <input_data_file> -l <input_labal_file> -o <output_directory> [optional args]")

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

    DATA_PATH  = args['testData']
    LABEL_PATH = args['testLabels']
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
        print()
    elif ( CLASSIFIER == LOG ):
        #TODO : Implement Logistic Classifier
        print()
    elif ( CLASSIFIER == RAF ):
        #TODO : Implement Random Forest Classifier
        print()
