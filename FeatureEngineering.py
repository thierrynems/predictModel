#author: Abid Hasan
# University of california Riverside
# April 2019

# importing packages
import sys
import time
import os
import pandas as pd
# import from other class
from AnalyzeParameter import *
from FileProcessing import *
from FeatureGeneration import *
from DatasetProcessing import *
from DNNModel import *
from FeatureGenerationSeq import *
#import the use library 
from keras.models import model_from_json
from tensorflow import keras
from keras.models import load_model
#from tensorflow.keras.models import load_model
# in tool parameter
paramDict = {
    'bins': 1,
    'trainingProp': 0.8, 
    'repeat': 10
}
optimizerDict = {
    'adadelta': Adadelta(lr=1.69, rho=0.95, epsilon=0.00001, decay=0.0),
}
# parameter dictionary
paramDictModel = {
    'epoch': 50,
    'batchSize': 256,
    'dropOut': 0.25,
    'loss': 'binary_crossentropy',
    'metrics': ['accuracy'],
    'activation1': 'relu',
    'activation2': 'sigmoid',
    'monitor': 'val_acc',  # param for checkpoint
    'save_best_only': True,
    'mode': 'max'
}

#datasetNT = sys.argv[1]
#datasetAA = sys.argv[2]
datasetNT = "input/seqGene.fasta"
datasetAA = "input/seqProtein.fasta"

#proportion 'trainingProp': 0.8,
#'trainingProp': 0.8, //proportion data d'entrainement 

#Changed correlation from 0.9 to 0.95, dropout changed from 0.3 to 0.2, learning rate from 1 to 1.645, and topology to 256-512-1024-512-1024-512
#########################################################
# returns the feature table containing attributed of each gene. The decision of
# returning essential/non essential gene feature table is made with class label parameter
def getGeneFeatTableSeq(feat, geneName,classLabel=1):
    featList = list()
    EssentialGeneLengthDict = feat.getEssentialGeneLengthFeatDict()
    EssentialKmerFeatDict = feat.getEssentialKmerFeatDict()
    EssentialGCFeatDict = feat.getEssentialGCContentFeatDict()
    EssentialCIARCSUFeatDict = feat.getEssentialCAIRCSUFeatDict()
    EssentialProteinFeatDict = feat.getEssentialProteinFeatDict()
    featList.append(EssentialGeneLengthDict[geneName])
    featList.extend(EssentialKmerFeatDict[geneName])
    featList.extend(EssentialGCFeatDict[geneName])
    featList.extend(EssentialCIARCSUFeatDict[geneName])
    featList.extend(EssentialProteinFeatDict[geneName])
    #featList.append(classLabel)  # 
    #featList.append(attributeList)
            
    return featList

def getScaledData(dataMatrix):
        scaler = StandardScaler().fit(dataMatrix[:, np.newaxis])
        return scaler.transform(dataMatrix[:, np.newaxis])
        #return scaler.transform(dataMatrix)

#########################################################
def main():
    essential_dir_path = "input/essential/"
    non_essential_dir_path  = "input/nonessential/"
    mcl_file_path = "orthoMCL.txt"
    dataset = "dataset.txt"
    option = "-c"
    experimentName = "output/"+ "unmodifiedResults"
    
   
    # program start time
    start_time = time.time()

    # processing parameters
    param = ParameterProcessing(essential_dir_path, non_essential_dir_path, dataset, mcl_file_path)
    # read the files and extract information from the file
    read = ReadFiles(param)
    read.getDatasetFileInfo()  # read the dataset for other method to access dataset info
    
    # process the files and generate features from sequence and other files
    #self, read,geneSeq,protSeq

    #datasetAA="input/nonEssential/degaa-np.dat"
    #datasetNT="input/nonEssential/degseq-np.dat"
    OrganismName="deeplyFeatureEngineering"
    #column of matrix of feature
    columnFeat=['NT-Length','TTT', 'TTC', 'TTA', 'TTG', 'CTT','CTC', 'CTA', 'CTG', 'ATT', 'ATC','ATA', 'ATG', 'GTT', 'GTC', 'GTA','GTG', 'TAT', 'TAC', 'TAA', 'TAG','CAT', 'CAC', 'CAA', 'CAG', 'AAT','AAC', 'AAA', 'AAG', 'GAT', 'GAC','GAA', 'GAG', 'TCT', 'TCC', 'TCA','TCG', 'CCT', 'CCC', 'CCA', 'CCG','ACT', 'ACC', 'ACA', 'ACG', 'GCT','GCC', 'GCA', 'GCG', 'TGT', 'TGC','TGA', 'TGG', 'CGT', 'CGC', 'CGA','CGG', 'AGT', 'AGC', 'AGA', 'AGG','GGT', 'GGC', 'GGA', 'GGG','GC','CIA','RCSU','A', 'R', 'N', 'D','C', 'Q', 'E', 'G','H', 'I', 'L', 'K','M', 'F', 'P', 'S','T', 'W', 'Y', 'V','AA-Length'];
    #Org="dde"
    #df = pandas.read_table(dataset,sep= '\t', header='infer')

    #baseDir="output/"
    #if os.path.isdir(baseDir): 
    #	print("le repertoire existe")
    #else: 
    #	print("Creation du repertoire "+ baseDir)
    #	os.mkdir(baseDir);

    filename=str(OrganismName)+".csv"
    #file = open(filename, "w")
    fastaSequencesNT = SeqIO.parse(open(datasetNT), 'fasta')
    fastaSequencesAA = SeqIO.parse(open(datasetAA), 'fasta')
    featureSeq=list()
    geneList=list()
    for fastaNT in fastaSequencesNT:
        nameNT, sequenceNT = fastaNT.id, str(fastaNT.seq)
        isSequence=0
        idSequenceNN=nameNT
        for fastaAA in fastaSequencesAA: 
            nameAA, sequenceAA = fastaAA.id, str(fastaAA.seq)
            idSequenceAA=nameAA
            if idSequenceAA == idSequenceNN :
                isSequence=1
                break;
        #print("NT: "+idSequenceNN+" AA: "+idSequenceNN)
        if isSequence == 1: # les deux sequence existe 
            sequenceNT = sequenceNT.upper()
            sequenceAA = sequenceAA.upper()
            geneList.append(nameNT) 
            feat = FeatureProcessingSeq(read,sequenceNT,sequenceAA)
            feat.getFeaturesSeq(paramDict['bins'],nameNT)
            #combiner toutes les features
            dataToPredict=getGeneFeatTableSeq(feat, nameNT)
            featureSeq.append(dataToPredict)
            #myData=np.array(dataToPredict)
            #myData.append(dataToPredict[:])
            ####################################
            #mydata=np.asarray(dataToPredict)
            #mydata=mydata.reshape(1,-1) 
            #print(myData)
            #sys.exit()
            ################################""""""
            #myData=myData.reshape(1,-1) 
            #dataMatrix=getScaledData(myData)
            #dataMatrix=dataMatrix.reshape(1,-1)
            #dataMatrix=dataMatrix.ravel()
            #print("#########################################")
            #print(dataMatrix)
            #print("#########################################")
            #sys.exit()
            ##model de prediction 
            ################################Old load model 3333 #######################################
    #file.close()
    myData=np.array(featureSeq, dtype='f')
    #df = pd.DataFrame(data = numpyArray, index = ["Row_1", "Row_2"], columns = ["Column_1","Column_2", "Column_3"]) 
    df = pd.DataFrame(data = myData,index = geneList , columns = columnFeat) 
    df.to_csv(filename, index=True)
    #print(geneList)
    sys.exit()
   
if __name__ == "__main__":
    main()
