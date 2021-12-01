# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 11:19:52 2021

@author: Danylo
"""

#%reset -f



#%% Function takes in string and outputs estimated time to complete and units
import pickle
import numpy as np
from datetime import datetime,timedelta
import math

# load model
model_filename= "model_RandomForestRegressor.pkl"
with open(model_filename, 'rb') as infile:
    model_REG=pickle.load(infile)

# load vectorizer    
vectorizer_filename= "vectorizer_count.pkl"
with open(vectorizer_filename, 'rb') as infile:
    vectorizer=pickle.load(infile)
 
# load vectorized X    
X_filename= "vectorizer_count_X.pkl"
with open(X_filename, 'rb') as infile:
    X=pickle.load(infile)    
 
    
# do the same as above for predicting units
# load model
model_filename= "model_RandomForestClassifier.pkl"
with open(model_filename, 'rb') as infile:
    model_CLS=pickle.load(infile)

# load Label Encoder    
lenc_filename= "lenc_target.pkl"
with open(lenc_filename, 'rb') as infile:
    lenc=pickle.load(infile)
    
# load dataframe
dataframe_filename = "dataframe.pkl"    
with open(dataframe_filename, 'rb') as infile:
    df=pickle.load(infile)
        
        


#%% CM_Predict_DPU
def CM_Predict_DPU(inputstring):
    
    # pedict rate value, days per unit
    pred_dpu = model_REG.predict(vectorizer.transform([inputstring]))
    
    return pred_dpu

#%% CM_Predict_Units
def CM_Predict_Units(inputstring):
    
    # pedict unit type
    _pred_unit = model_CLS.predict(vectorizer.transform([inputstring]))
    _pred_unit_str = lenc.inverse_transform(_pred_unit.ravel())
    pred_unit = " "
    pred_unit = ' '.join(map(str,_pred_unit_str))
       
    return pred_unit

#%% CM_SearchForMatch
def CM_SearchForMatch(inputstring):
    # search for string
    _a = vectorizer.transform([inputstring])
    _b = X.dot(_a.transpose())
    _stringmatches = np.where(_b.todense() == int(max(_b.todense())))
    stringmatches = _stringmatches[0]

    return stringmatches


#%% Days per Unit - Lowest From Search
def CM_SearchLow(inputstring):
    stringmatches = CM_SearchForMatch(inputstring)
    _a = df['COMPLETION RATE DAYS PER UNIT'].values
    search_min_dpu = min(_a[stringmatches])
    return search_min_dpu

#%% Days per Unit - Highest From Search
def CM_SearchHigh(inputstring):
    stringmatches = CM_SearchForMatch(inputstring)
    _a = df['COMPLETION RATE DAYS PER UNIT'].values
    search_max_dpu = max(_a[stringmatches])
    return search_max_dpu

#%% CM_Predict
def CM_Predict(Quantity,Transformation,Description,StartDate,EndDate):
    screen_output = []
    print(len(Transformation))
    for i_trans in range(len(Transformation)):
        inputstring = Transformation[i_trans] + ' ' + Description[i_trans]
        
        # Error handing, missing info
        if inputstring == 'Other' or inputstring == ' ':
            print('Please add additional information')
        
        pred_dpu = CM_Predict_DPU(inputstring)
        search_min_dpu = CM_SearchLow(inputstring)
        search_max_dpu = CM_SearchHigh(inputstring)
        if pred_dpu < search_min_dpu:
            pred_dpu = search_min_dpu
        if pred_dpu > search_max_dpu:
            pred_dpu = search_max_dpu   
        
        # Scenario 1 - Quantity yes, startdate yes, enddate yes
        if Quantity[i_trans] != [] and StartDate != '' and EndDate != '':

            print('========== Prediction for Activity',int(i_trans+1),'==========')
            print('Predicted Completion Rate, Days per Unit: {:.3f} (min: {:.3f}, max: {:.3f})'.format(float(pred_dpu),float(search_min_dpu),float(search_max_dpu)))
            print('Predicted Completion Rate, Units per Day: {:.3f} (min: {:.3f}, max: {:.3f})'.format(1/float(pred_dpu),1/float(search_max_dpu),1/float(search_min_dpu)))
            pred_unit = CM_Predict_Units(inputstring)
            print('Units:',pred_unit)
            print('Quantity:',Quantity[i_trans])
            print('Start Date:',StartDate)
            print('End Date:',EndDate)
            
            days_avail = datetime.strptime(EndDate,'%Y-%m-%d') - datetime.strptime(StartDate,'%Y-%m-%d')
            pred_numdays = math.ceil(Quantity[i_trans]*pred_dpu)
            newEndDate = datetime.strptime(StartDate,'%Y-%m-%d') + timedelta(days=pred_numdays)
            if Quantity[i_trans] < (1/float(pred_dpu))*(days_avail/timedelta(days=1)):
                print('--> Sufficient Time Allocated to Project')
                #print('Number of Units that could be completed in timeframe: {:.3f}'.format((1/float(pred_dpu))*(days_avail/timedelta(days=1))))
                
                print('--> Predicted Activity End Date: {}'.format(datetime.strftime(newEndDate,'%Y-%m-%d')))
            else:
                print('--> Insufficient Time Allocated to Activity')
                print('--> Predicted Activity End Date: {}'.format(datetime.strftime(newEndDate,'%Y-%m-%d')))

        # Scenario 2 - Quantity yes, startdate yes, enddate no
        elif Quantity[i_trans] != [] and StartDate != '' and EndDate == '':
            print('========== Prediction for Activity',int(i_trans+1),'==========')
            print('Predicted Completion Rate, Days per Unit: {:.3f} (min: {:.3f}, max: {:.3f})'.format(float(pred_dpu),float(search_min_dpu),float(search_max_dpu)))
            print('Predicted Completion Rate, Units per Day: {:.3f} (min: {:.3f}, max: {:.3f})'.format(1/float(pred_dpu),1/float(search_max_dpu),1/float(search_min_dpu)))
            pred_unit = CM_Predict_Units(inputstring)
            print('Units:',pred_unit)
            print('Quantity:',Quantity[i_trans])
            print('Start Date:',StartDate)
            pred_numdays = math.ceil(Quantity[i_trans]*pred_dpu)
            newEndDate = datetime.strptime(StartDate,'%Y-%m-%d') + timedelta(days=pred_numdays)
            print('--> Predicted End Date: {}'.format(datetime.strftime(newEndDate,'%Y-%m-%d')))

        # Scenario 3 - Quantity yes, startdate no, enddate yes
        elif Quantity[i_trans] != [] and StartDate == '' and EndDate != '':
            print('========== Prediction for Activity',int(i_trans+1),'==========')
            print('Predicted Completion Rate, Days per Unit: {:.3f} (min: {:.3f}, max: {:.3f})'.format(float(pred_dpu),float(search_min_dpu),float(search_max_dpu)))
            print('Predicted Completion Rate, Units per Day: {:.3f} (min: {:.3f}, max: {:.3f})'.format(1/float(pred_dpu),1/float(search_max_dpu),1/float(search_min_dpu)))
            pred_unit = CM_Predict_Units(inputstring)
            print('Units:',pred_unit)
            print('Quantity:',Quantity[i_trans])
            print('End Date:',EndDate)
            pred_numdays = math.ceil(Quantity[i_trans]*pred_dpu)
            newStartDate = datetime.strptime(EndDate,'%Y-%m-%d') - timedelta(days=pred_numdays)
            print('--> To Complete Activity on Time, Start By Predicted Start Date: {}'.format(datetime.strftime(newStartDate,'%Y-%m-%d')))
        
        # Scenario 4 - Quantity yes, startdate no, enddate no
        elif Quantity[i_trans] != [] and StartDate == '' and EndDate == '':
            print('========== Prediction for Activity',int(i_trans+1),'==========')
            print('Predicted Completion Rate, Days per Unit: {:.3f} (min: {:.3f}, max: {:.3f})'.format(float(pred_dpu),float(search_min_dpu),float(search_max_dpu)))
            print('Predicted Completion Rate, Units per Day: {:.3f} (min: {:.3f}, max: {:.3f})'.format(1/float(pred_dpu),1/float(search_max_dpu),1/float(search_min_dpu)))
            pred_unit = CM_Predict_Units(inputstring)
            print('Units:',pred_unit)
            print('Quantity:',Quantity[i_trans])
            pred_numdays = math.ceil(Quantity[i_trans]*pred_dpu)
            print('--> Total Predicted Completion Time: {} days'.format(pred_numdays))
            
        # Scenario 5 - Quantity no, startdate yes, enddate yes
        elif not(bool(Quantity[i_trans])) and StartDate != '' and EndDate != '':
            print('========== Prediction for Activity',int(i_trans+1),'==========')
            print('Predicted Completion Rate, Days per Unit: {:.3f} (min: {:.3f}, max: {:.3f})'.format(float(pred_dpu),float(search_min_dpu),float(search_max_dpu)))
            print('Predicted Completion Rate, Units per Day: {:.3f} (min: {:.3f}, max: {:.3f})'.format(1/float(pred_dpu),1/float(search_max_dpu),1/float(search_min_dpu)))
            pred_unit = CM_Predict_Units(inputstring)
            print('Units:',pred_unit)
            print('Start Date:',StartDate)
            print('End Date:',EndDate)
            num_days_available = (datetime.strptime(EndDate,'%Y-%m-%d') - datetime.strptime(StartDate,'%Y-%m-%d'))/timedelta(days=1)
            pred_units_completed = num_days_available/pred_dpu
            print('--> Predicted Number of Units Completed in Timeframe: {:.3f}'.format(float(pred_units_completed)))
        # Scenario 6,7,8 - Quantity no, startdate no OR enddate no
        else:
            print('========== Prediction for Activity',int(i_trans+1),'==========')
            print('Predicted Completion Rate, Days per Unit: {:.3f} (min: {:.3f}, max: {:.3f})'.format(float(pred_dpu),float(search_min_dpu),float(search_max_dpu)))
            print('Predicted Completion Rate, Units per Day: {:.3f} (min: {:.3f}, max: {:.3f})'.format(1/float(pred_dpu),1/float(search_max_dpu),1/float(search_min_dpu)))
            pred_unit = CM_Predict_Units(inputstring)
            print('Units:',pred_unit)
            print('(Please add additional information (quantity, start date, end date) for additional analysis)')
        

    '''
    # print values
    print('Days per unit: {:.3f}'.format(float(pred_dpu)))
    print('Units per Day: {:.3f}'.format(1/float(pred_dpu)))
    print('Units:',pred_unit)
    '''
     
        
    return


#%% Sample Inputs For Testing 
Quantity_1 = 25554
Quantity_1 = []
Transformation_1 = 'form work'
Description_1 = 'plinth'
StartDate = '2021-05-06'
#StartDate = ''
EndDate = '2022-07-22'
EndDate = ''

inputstring_1= Transformation_1 + ' ' + Description_1


Quantity = [Quantity_1]
Transformation = [Transformation_1]
Description = [Description_1]

CM_Predict(Quantity,Transformation,Description,StartDate,EndDate)