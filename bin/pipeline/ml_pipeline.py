# -*- coding: utf-8 -*-
"""Main module."""

from Transformer import CategoricalTransformer 
from Transformer import NumericalTransformer 
from Transformer import FeatureSelector 
from constants import *
import pandas as pd
import os
import sys
sys.path.append('bin/')

#sklearn
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import FeatureUnion, Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')
    
def XGBoost_Model_Predict():    
    
    
    train = pd.read_csv('datasets/ML_Train.csv',low_memory=False)
    print(train.shape)
    #You can covert the target variable to numpy 
    
    X_train = train[train.columns.difference(['y'])] 
    y_train = train[['y']]
    
    test = pd.read_csv('datasets/ML_Test.csv',low_memory=False)
    X_test = test[test.columns.difference(['y'])] 
    y_test = test[['y']]   
    
    #Defining the steps in the categorical pipeline 
    categorical_pipeline1 = Pipeline( steps = [ ( 'cat_selector', FeatureSelector(categorical_features) ),                                    
                                    ( 'cat_transformer', CategoricalTransformer() ),                                   
                                    ( 'one_hot_encoder', OneHotEncoder( sparse = False ) ) ] )    
    
        
    #Defining the steps in the numerical pipeline     
    numerical_pipeline = Pipeline( steps = [ ( 'num_selector', FeatureSelector(numerical_features) ),                                    
                                    ( 'num_transformer', NumericalTransformer() ),       
                                    ( 'std_scaler', StandardScaler() ) ] )                                

    #Combining numerical and categorical piepline into one full big pipeline horizontally 
    #using FeatureUnion
    data_pipeline = FeatureUnion( transformer_list = [( 'numerical_pipeline', numerical_pipeline ), 
                                                      ( 'categorical_pipeline1', categorical_pipeline1 )                                                                                                              
                                                     ] )
    
    #The full pipeline as a step in another pipeline with an estimator as the final step
    ml_pipeline = Pipeline( steps = [ ( 'data_pipeline', data_pipeline),                                    
                                    ( 'model', XGBClassifier() ) ] )  
    
    
    y_train['y'] = y_train['y'].replace('no',0,regex=True)
    y_train['y'] = y_train['y'].replace('yes',1,regex=True)
    
    y_test['y'] = y_test['y'].replace('no',0,regex=True)
    y_test['y'] = y_test['y'].replace('yes',1,regex=True)
    
    ml_pipeline.fit(X_train, y_train)
    
    #Can predict with it like any other pipeline
    y_pred = ml_pipeline.predict( X_test )   
    
    cm = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix: {}'.format(cm))
    print('Classification Report: {}'.format(classification_report(y_test, y_pred)))    
    
    y_pred_proba = ml_pipeline.predict_proba(X_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)    
    print('Best AUC Score: {}'.format(auc))
    
    return round(auc,4)

out = XGBoost_Model_Predict()
print(out)


