import sys
import numpy as np

# Initialising variables
arrayCGHData = ""
arrayGroups = ""
cghData = []
groupDataDict = {}
compiledData = []
cghDataList = []
cghDataDict = {}

# Get the files from the input or use default
try:
    arrayCGHData = sys.argv[1]
    arrayGroups = sys.argv[2]
except IndexError:
    print("No data files given. I will use the default files!!")
    arrayCGHData = "Train_call.txt"
    arrayGroups = "Train_Clinical.txt"

fileData = open(arrayCGHData, "r")
fileGroups = open(arrayGroups, "r")

# Get the data from CGH Data file
for line in fileData:
    lineTerm = line.rstrip("\n").split("\t")
    cghData.append(lineTerm)
fileData.close()

# Remove the unwanted data and transpose the CGH Data matrix
cghDataMatrix = np.matrix(cghData)
cghDataMatrixTrans = cghDataMatrix.transpose()
#cghDataMatrixTrans = np.delete(cghDataMatrixTrans , [1,2,3], 0)
cghDataList = cghDataMatrixTrans.tolist()
#print(cghDataList)

#print(cghDataList[1][1:])

# Store the CGH Data into a dictionary
for i in range(0,101):
    a = cghDataList[i][0]
    array = a.replace('"',"")
    cghDataDict[array] = cghDataList[i][1:]

#print(cghDataDict['Chromosome'])
# Get the data from the array groups file
for line in fileGroups:
    lineTerm = line.rstrip("\n").split("\t")
    lineTerm[0] = lineTerm[0].replace('"',"")
    groupDataDict[lineTerm[0]] = lineTerm[1]
fileGroups.close()

# Compile the final matrix - merge the data from the dictionaries based on Array

colHeader = []
for i in range(1,len(cghDataList[0])):
    #string = 'cghDataList[1][i], " - ", cghDataList[2][i]'
    string = str(cghDataList[1][i]) + " - " + str(cghDataList[2][i])
    colHeader.append(string)
    
compiledData.append(["Array","Group", *colHeader])

for a,v in cghDataDict.items():
    if "Array" in a:
        #print(groupDataDict[a])
        compiledData.append([a, groupDataDict[a],*cghDataDict[a]])

# Print the final List into a file
outputFile = open("Complied-Data.txt","w")
for i in compiledData:
    for j in i:
        outputFile.write(j)
        outputFile.write("\t")
    outputFile.write("\n")
outputFile.close()
     
