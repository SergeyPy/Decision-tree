#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from function_classification import *
from function_regression import *


# In[1]:


class NapoleonRegression(object):
    """docstring"""
    
    def __init__(self, max_depth=2, min_samples=1, criterion='variance'):
        """Constructor"""
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.criterion = criterion

    
    
    def fit(self,x,y):
        """
        Fit model
        """
        if type(x) == pd.DataFrame or type(y) == pd.Series:
            x = x.values
            y = y.values
        
        root  = get_split(x,y,self.criterion) #делаем разбиение в корне, получаем  словарь содержащий индекс, значение и группу
        split(root,self.max_depth, self.min_samples, 1,self.criterion)
        self.root = root
        return self.root 

    
    def prediction(self,x):
        """
        """
        if type(x) == pd.DataFrame:
            x = x.values
            
        values=[]
        for row in x:
            pr = predict(self.root,row)
            values.append(pr)   
        return  values  


# In[3]:


class NapoleonClassifier(object):
    """docstring"""
    
    def __init__(self, max_depth=2, min_samples=1, criterion='Gini'):
        """Constructor"""
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.criterion = criterion

    
    
    def fit(self,x,y):
        """
        Fit model
        """
        x = x.values
        y = y.values
        
        root  = get_split_c(x,y,self.criterion) #делаем разбиение в корне, получаем  словарь содержащий индекс, значение и группу
        split_c(root,self.max_depth, self.min_samples, 1,self.criterion)
        self.root = root
        return self.root 

    
    def prediction(self,x):
        """
        """
        x =x.values
        values=[]
        for row in x:
            pr = predict_c(self.root,row)
            values.append(pr)   
        return  values  

