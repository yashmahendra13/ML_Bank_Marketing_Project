from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

#Custom Transformer that extracts columns passed as argument to its constructor 
class FeatureSelector( BaseEstimator, TransformerMixin ):
    #Class Constructor 
    def __init__( self, feature_names ):
        self._feature_names = feature_names 
    
    #Return self nothing else to do here    
    def fit( self, X, y = None ):
        return self 
    
    #Method that describes what we need this transformer to do
    def transform( self, X, y = None ):
        return X[ self._feature_names ]
    
class NumericalTransformer(BaseEstimator, TransformerMixin):
    #Class Constructor
    def __init__( self):
        pass
        
    #Return self, nothing else to do here
    def fit( self, X, y = None ):
        return self   
    
    #Log Transformation
    def Log_Transform(self,x):
        return np.log(self.x)
    
    #Custom transform method we wrote that creates aformentioned features and drops redundant ones 
    def transform(self, X, y = None):         
        
        X.loc[:,'campaign'] = X['campaign'].apply(lambda x : np.log(x))         
        
        #returns a numpy array
        return X.values

class CategoricalTransformer( BaseEstimator, TransformerMixin ):
    #Class constructor method that takes in a list of values as its argument
    def __init__(self):
        pass
        
    #Return self nothing else to do here
    def fit( self, X, y = None  ):
        return self

    #Define Function to Transform pdays to categorical feature
    def feature_pdays(self,val):        
        if 0 <= val < 5:
            return 'pdays_0_5'
        elif 5 <= val < 10:
            return 'pdays_5_10'
        elif 10 <= val < 16:
            return 'pdays_10_16'
        elif 16 <= val < 21:
            return 'pdays_16_21'
        elif 21 <= val < 21:
            return 'pdays_21_27'
        elif val >=27:
            return 'pdays_no_contact'
        else:
            return 'pdays_missing'   
    
    #Transformer method we wrote for this transformer 
    def transform(self, X , y = None ):       
          
       #Convert pdays to categorical variable       
       X.loc[:,'pdays'] = X['pdays'].apply(lambda x : self.feature_pdays(x))      
       
       return X.values 