# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 11:19:52 2021

@author: Danylo
"""

### Function takes in string and outputs estimated time to complete and units
import pickle

# load model
model_filename= "model_BaggingRegressor.pkl"
with open(model_filename, 'rb') as infile:
    model_REG=pickle.load(infile)

# load vectorizer    
vectorizer_filename= "vectorizer.pkl"
with open(vectorizer_filename, 'rb') as infile:
    vectorizer=pickle.load(infile)
    
# do the same as above for predicting units
# load model
model_filename= "model_RandomForestClassifier.pkl"
with open(model_filename, 'rb') as infile:
    model_CLS=pickle.load(infile)

# load Label Encoder    
lenc_filename= "lenc_target.pkl"
with open(lenc_filename, 'rb') as infile:
    lenc=pickle.load(infile)    
        
        

def CM_Predict(inputstring):
    
    
    # pedict value
    time_pred = model_REG.predict(vectorizer.transform([inputstring]))
    
    # pedict value
    _unit_pred = model_CLS.predict(vectorizer.transform([inputstring]))
    _unit_pred_str = lenc.inverse_transform(_unit_pred.ravel())
    unit_pred = " "
    unit_pred = ' '.join(map(str,_unit_pred_str))
    
      
    # print values
    print('Days per unit: {:.3f}'.format(float(time_pred)))
    print('Units per Day: {:.3f}'.format(1/float(time_pred)))
    print('Units:',unit_pred)
    #    return y_pred
    return None

