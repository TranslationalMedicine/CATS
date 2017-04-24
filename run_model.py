#!/usr/bin/env python3
"""Reproduce your result by your saved model.
This is a script that helps reproduce your prediction results using your saved
model. This script is unfinished and you need to fill in to make this script
work. If you are using R, please use the R script template instead.
The script needs to work by typing the following commandline (file names can be
different):
python3 run_model.py -i unlabelled_sample.txt -m model.pkl -o output.txt
"""

# author: Chao (Cico) Zhang
# date: 31 Mar 2017

import argparse
import sys
from sklearn.externals import joblib
import numpy as np
import pandas as pd

# Start your coding

def parse_samples(inputFile): 
    """ To parse the input file to required format."""
    # Reading in the file and transpose the data frame.
    data=pd.read_table(inputFile, sep='\t', delimiter=None, delim_whitespace=False, header=0, index_col=0)
    data = data.transpose()
    
    # Modifying the columns
    columnList = list(data.columns.values)
    newColumnNames = []
    for i in range(0,len(data.iloc[0])):
        newColumnNames.append(str(data.iloc[0,i]) + "-" + str(data.iloc[1,i]))
    data.columns=newColumnNames
    data.insert(0,"Groups","")
    data = data.drop(data.index[[0,1,2]])
    
    # Getting the required parts of the data
    i = len(data)
    j = len(data.iloc[0])
    X = (data.iloc[0:i, 1:j]) #NB: 150 is feature 1:149 for now, because of the long running time
    return (X)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Reproduce the prediction')
    parser.add_argument('-i', '--input', required=True, dest='input_file',
                        metavar='unlabelled_sample.txt', type=str,
                        help='Path of the input file')
    parser.add_argument('-m', '--model', required=True, dest='model_file',
                        metavar='model.pkl', type=str,
                        help='Path of the model file')
    parser.add_argument('-o', '--output', required=True,
                        dest='output_file', metavar='output.txt', type=str,
                        help='Path of the output file')
    # Parse options
    args = parser.parse_args()

    if args.input_file is None:
        sys.exit('Input is missing!')

    if args.model_file is None:
        sys.exit('Model file is missing!')

    if args.output_file is None:
        sys.exit('Output is not designated!')

    # Start your coding

    # Getting the values from arguement parser
    outputFile = args.output_file
    model_file = args.model_file
    inputFile = args.input_file
    
    # Parsing the input data
    X = parse_samples(inputFile)
        
    # Predictions using the model
    model = joblib.load(model_file) 
    predictions = (model.predict(X))
    predictions = predictions.tolist()

    # Changing output labels (0,1,2) to cancer type labels
    for i in range(0,len(predictions)):
        if predictions[i]==0:
            predictions[i]="HER2+"
        elif predictions[i]==1:
            predictions[i]="HR+"
        elif predictions[i]==2:
            predictions[i]="Triple negative"

    # Writing to output file			
    outputFile = open(outputFile,"w")
    outputFile.write('"Sample"' + "\t" + '"SubGroup"' + "\n")	
    for i in range(0,len(X.index)):
        array = '"' + X.index[i] + '"'
        predicted = '"' + predictions[i] + '"'
        outputFile.write(array + "\t" + predicted + "\n")
    outputFile.close()	
  
    # End your coding

if __name__ == '__main__':
    main()

