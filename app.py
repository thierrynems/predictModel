# author: Abid Hasan
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
from keras import backend as K
#Flask app
# app.py - a minimal flask api using flask_restful
from flask import Flask,jsonify,request
from flask_restful import Resource, Api
############################################################################################
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
def getGeneFeatProperties(feat, geneName,classLabel=1):
    featList = dict()
    EssentialGeneLengthDict = feat.getEssentialGeneLengthFeatDict()
    EssentialKmerFeatDict = feat.getEssentialKmerFeatDict()
    EssentialGCFeatDict = feat.getEssentialGCContentFeatDict()
    EssentialCIARCSUFeatDict = feat.getEssentialCAIRCSUFeatDict()
    EssentialProteinFeatDict = feat.getEssentialProteinFeatDict()
    featList={
        "GeneLength": EssentialGeneLengthDict,
        "KmerFeat": EssentialKmerFeatDict,
        "GCFeat": EssentialGCFeatDict,
        "CIARCSUFeat": EssentialCIARCSUFeatDict,
        "ProteinFeat":EssentialProteinFeatDict
    }
                 
    return featList

def getScaledData(dataMatrix):
        scaler = StandardScaler().fit(dataMatrix[:, np.newaxis])
        return scaler.transform(dataMatrix[:, np.newaxis])
        #return scaler.transform(dataMatrix)

####################FLASK API ###############################################################
app = Flask(__name__)
api = Api(app)

class HelloWorld(Resource):
    def get(self):
        return {'An approach of Machine Learning to predict essential gene : ': 'Deeply Essential'}

class ReceiveGene(Resource):
  def post(self):
    #get posted data from request
    geneSeq=request.form.get('geneSeq')
    protSeq=request.form.get('protSeq')
    #convertion en string
    geneSeq=str(geneSeq)
    protSeq=str(protSeq)
     
    if(geneSeq!="" and protSeq!=""):
        essential_dir_path = "input/essential/"
        non_essential_dir_path  = "input/nonessential/"
        mcl_file_path = "orthoMCL.txt"
        dataset = "dataset.txt"
        option = "-c"
        experimentName = "output/"+ "unmodifiedResults"
        
        # processing parameters
        param = ParameterProcessing(essential_dir_path, non_essential_dir_path, dataset, mcl_file_path)
        # read the files and extract information from the file
        read = ReadFiles(param)
        read.getDatasetFileInfo()  # read the dataset for other method to access dataset info
        
        # process the files and generate features from sequence and other files
        #self, read,geneSeq,protSeq
        seqName="DEG100011"
        #essential Gene
                            
        feat = FeatureProcessingSeq(read,geneSeq,protSeq)
        feat.getFeaturesSeq(paramDict['bins'])
       #combiner toutes les features
        dataToPredict=getGeneFeatTableSeq(feat, seqName)
        myData=np.array(dataToPredict)
        #print(mydata)
        #sys.exit()
        dataMatrix=getScaledData(myData)
        dataMatrix=dataMatrix.reshape(1,-1)
        ##model de prediction 
        json_file = open('deeply-model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load best model and compile
        loaded_model.load_weights('weights.best.hdf5')
        # compile model with a best parameter
        loaded_model.compile(optimizer=optimizerDict['adadelta'],
                    loss=paramDictModel['loss'],
                    metrics=paramDictModel['metrics'])
        #Before prediction
        #K.clear_session()
        # make a prediction
        prediction=loaded_model.predict(dataMatrix)
        #############################################
        #prediction=loaded_model.predict_classes(mydata)
        #testLabelRev = np.argmax(resultPrediction, axis=1)
        #print(testLabelRev)
        #probs = loaded_model.predict_proba(mydata)
        #print(probs)
        #sys.exit()

        #After prediction
        K.clear_session()
        retJson={
            "geneSeq":geneSeq,
            "protSeq":protSeq,
            "EssentialGeneProbability":str(prediction[:,1]),
            "NonEssentialGeneProbability":str(prediction[:,0])
        }
        return jsonify(retJson)
    else:
       retJson={
            "error":"les champs de sequence de gene et de proteins sont requis pour la predicition",
        } 
class TestGene(Resource):
  def post(self):
    #get posted data from request
    geneSeq=request.form.get('geneSeq')
    protSeq=request.form.get('protSeq')
    if(geneSeq!="" and protSeq!=""):
        essential_dir_path = "input/essential/"
        non_essential_dir_path  = "input/nonessential/"
        mcl_file_path = "orthoMCL.txt"
        dataset = "dataset.txt"
        option = "-c"
        experimentName = "output/"+ "unmodifiedResults"
        
        # processing parameters
        param = ParameterProcessing(essential_dir_path, non_essential_dir_path, dataset, mcl_file_path)
        # read the files and extract information from the file
        read = ReadFiles(param)
        read.getDatasetFileInfo()  # read the dataset for other method to access dataset info
        
        # process the files and generate features from sequence and other files
        #self, read,geneSeq,protSeq
        seqName="DEG100011"
        #essential Gene
                            
        feat = FeatureProcessingSeq(read,geneSeq,protSeq)
        feat.getFeaturesSeq(paramDict['bins'])
        #cget properties 
        geneProperties=getGeneFeatProperties(feat, seqName)

        return jsonify(geneProperties)
     

api.add_resource(HelloWorld, '/')
api.add_resource(ReceiveGene, '/deeply/prediction')
api.add_resource(TestGene, '/deeply/feature')
#app.run(debug=True, host='0.0.0.0', port=80)
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
