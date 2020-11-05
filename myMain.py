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
modelFile = int(sys.argv[3])

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
    OrganismName="OrganismDeeply"
    #Org="dde"
    #df = pandas.read_table(dataset,sep= '\t', header='infer')

    #baseDir="output/"
    #if os.path.isdir(baseDir): 
    #	print("le repertoire existe")
    #else: 
    #	print("Creation du repertoire "+ baseDir)
    #	os.mkdir(baseDir);

    filename=str(OrganismName)+".csv"
    file = open(filename, "a")
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
            myData=np.array(dataToPredict)
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
            print("argmax= "+str(testPredLabelRev))
            print("\n")
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
    # get features and build training/testing dataset
    data = ProcessData(read, feat, paramDict['trainingProp'], option, experimentName)

    # creating file to store evaluation statistics
    fWrite = open(experimentName + '.tab', 'w')
    fWrite.write("Experiment Name: " + str(experimentName) + '\n')
    fWrite.write("Number of training samples: " + str(data.getTrainingData().shape[0]) + '\n')
    fWrite.write("Number of validation samples: " + str(data.getValidationData().shape[0]) + '\n')
    fWrite.write("Number of testing samples: " + str(data.getTestingData().shape[0]) + '\n')
    fWrite.write("Number of features: " + str(data.getTrainingData().shape[1]) + '\n')
    fWrite.write("Iteration" + "\t" + "ROC_AUC" + "\t" + "Avg. Precision" + "\t" +
                 "Sensitivity" + "\t" + "Specificity" + "\t" + "PPV" + "\t" + "Accuracy" + "\n")

    # dict to store evaluation statistics to calculate average values
    evaluationValueForAvg = {
        'roc_auc': 0.,
        'precision': 0.,
        'sensitivity': 0.,
        'specificity': 0.,
        'PPV': 0.,
        'accuracy': 0.
    }

    # build DNN model
    if os.path.exists(experimentName + 'True_positives.txt'):
        os.remove(experimentName + 'True_positives.txt')
    if os.path.exists(experimentName + 'False_positives.txt'):
        os.remove(experimentName + 'False_positives.txt')
    if os.path.exists(experimentName + 'Thresholds.txt'):
        os.remove(experimentName + 'Thresholds.txt')

    f_tp = open(experimentName + 'True_positives.txt', 'a')
    f_fp = open(experimentName + 'False_positives.txt', 'a')
    f_th = open(experimentName + 'Thresholds.txt', 'a')

    for i in range(0, paramDict['repeat']):
        print('Iteration', i)
        model = BuildDNNModel(data, paramDict['bins'], f_tp, f_fp, f_th)
         
        evaluationDict = model.getEvaluationStat()

        print(evaluationDict)

        writeEvaluationStat(evaluationDict, fWrite, i + 1)

        evaluationValueForAvg['roc_auc'] += evaluationDict['roc_auc']
        evaluationValueForAvg['precision'] += evaluationDict['precision']
        evaluationValueForAvg['sensitivity'] += evaluationDict['sensitivity']
        evaluationValueForAvg['specificity'] += evaluationDict['specificity']
        evaluationValueForAvg['PPV'] += evaluationDict['PPV']
        evaluationValueForAvg['accuracy'] += evaluationDict['accuracy']

    for value in evaluationValueForAvg:
        evaluationValueForAvg[value] = float(evaluationValueForAvg[value]) / paramDict['repeat']

    writeEvaluationStat(evaluationValueForAvg, fWrite, 'Avg.')
    fWrite.write("\n")
    fWrite.write('Batch size:' + str(evaluationDict['batch_size']) + '\n')
    fWrite.write('Activation:' + str(evaluationDict['activation']) + '\n')
    fWrite.write('Dropout:' + str(evaluationDict['dropout']) + '\n')

    end_time = time.time()
    fWrite.write("Execution time: " + str(end_time - start_time) + " sec.")
    fWrite.close()
    # f_imp.close()

    f_tp.close()
    f_fp.close()
    f_th.close()


# writes the evaluation statistics
def writeEvaluationStat(evaluationDict, fWrite, iteration):
    fWrite.write(str(iteration) + "\t" + str(evaluationDict['roc_auc']) + "\t" +
                 str(evaluationDict['precision']) + '\t' + str(evaluationDict['sensitivity']) + '\t' +
                 str(evaluationDict['specificity']) + '\t' + str(evaluationDict['PPV']) + '\t' +
                 str(evaluationDict['accuracy']) + '\n')


if __name__ == "__main__":
    main()
