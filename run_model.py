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

# import the library you need here
def import_samples(): 
	#input for this should be unlabelled data file
	data=pd.read_table('Complied-Data.txt', sep='\t', delimiter=None, delim_whitespace=False, header=0, index_col=0)
	X = (data.iloc[0:100, 1:2835]) #NB: 150 is feature 1:149 for now, because of the long running time
	return (X)

# End your coding


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
    # suggested steps
    # Step 1: load the model from the model file
    # Step 2: apply the model to the input file to do the prediction
    # Step 3: write the prediction into the desinated output file
    output=str(sys.argv[3])	
    model_file=str(sys.argv[2])
    X=import_samples()
    model = joblib.load('model.pkl') 
    predictions=(model.predict(X))
    predictions=predictions.tolist()

    #changes output labels (0,1,2) to cancer type labels
    for i in range(0,len(predictions)):
    	if predictions[i]==0:
    		predictions[i]="HER2+"
    	elif predictions[i]==1:
    		predictions[i]="HR+"
    	elif predictions[i]==2:
    		predictions[i]="Triple negative"

    #writing to output file			
    outputFile = open(output,"w")		
    for i in range(0,len(X.index)):
    	outputFile.write(X.index[i] + "\t" + predictions[i] + "\n")
    	print (X.index[i], predictions[i])
    outputFile.close()	
   

    
    
    

    # End your coding


if __name__ == '__main__':
    main()
