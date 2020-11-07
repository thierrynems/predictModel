#author: Abid Hasan
# University of california Riverside
# April 2019

# importing packages
import sys
import time
import os

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

datasetNT = sys.argv[1]
datasetAA = sys.argv[2]
modelFile = int(sys.argv[3])-1

#list des modeles
listModel=("deeplyTrain9deeply-model.h5","sgsTrain9deeply-model.h5","deepHE9deeply-model.h5")
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
    OrganismName="deeplyPredictResult"
    #Org="dde"
    #df = pandas.read_table(dataset,sep= '\t', header='infer')

    #baseDir="output/"
    #if os.path.isdir(baseDir): 
    #	print("le repertoire existe")
    #else: 
    #	print("Creation du repertoire "+ baseDir)
    #	os.mkdir(baseDir);

    filename=str(OrganismName)+".csv"
    file = open(filename, "w")
    fastaSequencesNT = SeqIO.parse(open(datasetNT), 'fasta')
    fastaSequencesAA = SeqIO.parse(open(datasetAA), 'fasta')
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
            feat = FeatureProcessingSeq(read,sequenceNT,sequenceAA)
            feat.getFeaturesSeq(paramDict['bins'],nameNT)
            #combiner toutes les features
            dataToPredict=getGeneFeatTableSeq(feat, nameNT)
            #myData=np.array(dataToPredict)
            #myData.append(dataToPredict[:])
            ####################################
            #mydata=np.asarray(dataToPredict)
            #mydata=mydata.reshape(1,-1) 
            #print(myData)
            #sys.exit()
            ################################""""""
            #myData=myData.reshape(1,-1) 
            dataMatrix=getScaledData(myData)
            dataMatrix=dataMatrix.reshape(1,-1)
            #dataMatrix=dataMatrix.ravel()
            #print("#########################################")
            #print(dataMatrix)
            #print("#########################################")
            #sys.exit()
            ##model de prediction 
            ################################Old load model 3333 #######################################
            
            modelPath="ModelTrain/"+listModel[modelFile]
            loaded_model = load_model(modelPath)
            resultPrediction=loaded_model.predict(dataMatrix)
	   # print(resultPrediction.shape)
	   # sys.exit()
            E_score=resultPrediction[0][1]
            NE_score=resultPrediction[0][0]
            if E_score>=0.5: 
                decision="E"
            else: 
                decision="NE"
            #loaded_model.predict_classes
            testPredLabelRev = np.argmax(resultPrediction, axis=1)
            #print("argmax= "+str(testPredLabelRev))
            #print("\n")
            line= str(idSequenceNN)+";"+str(E_score)+";"+str(NE_score)+";"+str(decision)+"\n"
            print(line)
            file.write(str(line))
            ######################################################################
            #prediction=loaded_model.predict_classes(dataMatrix)
            #print(prediction)
            #probs = loaded_model.predict_proba(dataMatrix)
            #print(probs[:, 1])

            ###################################################################
            #testLabelRev = np.argmax(resultPrediction, axis=1)
            #print(testLabelRev)
            #probs = loaded_model.predict_proba(mydata)
            #print(probs)
    file.close()
    sys.exit()
   
if __name__ == "__main__":
    main()
