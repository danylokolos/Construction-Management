# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 11:19:52 2021

@author: Danylo
"""

### Function takes in string and outputs estimated time to complete



def CM_Predict(inputstring):
    import pickle
    
    
    # preprocess input string just like before model creation
    
    # load model
    model_filename= "model_BaggingRegressor.pkl"
    with open(model_filename, 'rb') as infile:
        model=pickle.load(infile)
    
    # load vectorizer    
    vectorizer_filename= "vectorizer.pkl"
    with open(vectorizer_filename, 'rb') as infile:
        vectorizer=pickle.load(infile)

    # pedict value
    y_pred = model.predict(vectorizer.transform([inputstring]))
    
    
    
    # do the same as above for predicting units
    
    
    print('Days per unit: {:.3f}'.format(float(y_pred)))
    print('Units per Day: {:.3f}'.format(1/float(y_pred)))

    #    return y_pred
    return None

