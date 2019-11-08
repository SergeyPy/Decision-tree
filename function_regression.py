#!/usr/bin/env python
# coding: utf-8

# numpy and pandas for data manipulation
import pandas as pd
import numpy as np



def test_split(index, value, x,y):
    left,left_y, right, right_y = list(), list(),list(), list() #инициализация left,left_y, right, right_y

    for i,row in enumerate(x): #  перебор по строкам
        tf = row[index] < value
        if tf: # если в строке значение по index  меньше value
            
            left.append(row)
                # Добавляем строку в левое поддерево
            left_y.append(y[i])  # Добавляем метки в левое поддерево
        else:
            right.append(row)# Добавляем строку в правое поддерево
            right_y.append(y[i])# Добавляем метки в правое поддерево
 
    
    return left,left_y, right,right_y





def variance(left,left_y, right,right_y):
    Xm = len(left)+len(right) #Вычесляем общее количество эдементов в левом и правом поддереве
    variance = 0.0 # началбная оценка variance
    for i,subtree in enumerate([left,right]):
        size_LR = len(subtree)
        if size_LR == 0:
            continue
        c = size_LR/Xm
        Q_score = 0.0
        Hx=0
        if i == 0:
            score=0
            y_size = len(left_y)
            y_sum = sum(left_y)
            if y_size == 0:
                continue
            y_ = y_sum/y_size
            count = 0
            for yi in left_y:
                score += (yi - y_)**2 #Дисперсия на левом поддереве
                count+=1    
        else:
            score=0
            y_size = len(right_y)
            y_sum = sum(right_y)
            if y_size == 0:
                continue
            y_ = y_sum/y_size
            count = 0
            for yi in right_y:
                score += (yi - y_)**2#Дисперсия на правом поддереве
                count+=1     
        Hx += (score/count)*c  
    return Hx


# In[6]:


def median(left,left_y, right,right_y):
    Xm = len(left)+len(right) #Вычесляем общее количество эдементов в левом и правом поддереве
    variance = 0.0 # началбная оценка variance
    for i,subtree in enumerate([left,right]):
        size_LR = len(subtree)
        if size_LR == 0:
            continue
        c = size_LR/Xm
        Q_score = 0.0
        Hx=0
        if i == 0:
            score=0
            y_size = len(left_y)
            if y_size == 0:
                continue
            mm = np.median(right_y)    
            for yi in left_y:
                score += (yi - mm)   #Median absolute mean
        else:
            score=0
            y_size = len(right_y)
            y_sum = sum(right_y)
            if y_size == 0:
                continue
            y_ = y_sum/y_size
            mm = np.median(right_y)
            for yi in right_y:
                score += (yi - mm)  #Median absolute mean    
        Hx += (score/y_size)*c    
    return Hx


# In[7]:



    
# Select the best split point for a dataset
def get_split(x,y,criterion):
    left,left_y, right, right_y = list(), list(),list(), list() 

    score = 0 #Начальная оценка
    b_score = 100
    for index in range(len(x[0])): #Перебор по индексам строки
        
        for row in  x:#

            left,left_y, right,right_y = test_split(index, row[int(index)], x,y) # вызов функции разбиения с параметрами [индекс, условие разбиения, набор данных x и y]; 
                                         # возвращает left_row и right_row
           
            if criterion =='variance':  
                score_c = variance(left,left_y, right,right_y ) # Оценка расщепления variance
                
            elif criterion =='median':
                score_c = median(left,left_y, right,right_y ) # Оценка расщепления median
                
            if score_c < b_score:
                b_score = score_c
                node = {'index':index,'values':row[int(index)],'l_subtree':left,'l_subtree_y':left_y,'r_subtree':right,'r_subtree_y':right_y}   #Узел           
    return node


# In[9]:


def to_terminal(left_y=[], right_y=[]): #Функция для расчета предсказания в листе
    outcomes = left_y+right_y
    return np.mean(outcomes)


# In[10]:


# Create child splits for a node or make terminal

def split(node,max_depth, min_samples, depth,criterion):
    left, right = node['l_subtree'], node['r_subtree']
    left_y, right_y = node['l_subtree_y'], node['r_subtree_y'] 

    #Проверка на пустые поддеревья
    if not left or not right:
        node['l_subtree'] = node['r_subtree'] = to_terminal(left_y , right_y)
        return
    
    
    # Проверка на максимальную глубину
    if depth >= max_depth:
        #Если юольше то расчитываем значени я в листях
        node['l_subtree'], node['r_subtree'] = to_terminal(left_y), to_terminal(right_y)
        return
    
    # Проверка на минимальное количество примеров в листе левого поддерева
    if len(left) <= min_samples:
        #Если меньше или равно, то расчитываем значени я в листях
        node['l_subtree'] = to_terminal(left_y,right_y)
    else:
        #Иначе продплжаем расщеплять
        node['l_subtree'] = get_split(left,left_y,criterion)
        split(node['l_subtree'],max_depth, min_samples, depth+1,criterion)
        
        
    # Проверка на минимальное количество примеров в листе правого поддерева
    if len(right) <= min_samples:
        node['r_subtree'] = to_terminal(left_y,right_y)
    else:
        node['r_subtree'] = get_split(right,right_y,criterion)
        split(node['r_subtree'], max_depth, min_samples, depth+1,criterion)


# In[11]:


def predict(node, row): #Функция для предсказания значений
    tf = row[node['index']] < node['values']
    if tf:
        if isinstance(node['l_subtree'], dict):
            return predict(node['l_subtree'], row)
        else:
            return node['l_subtree']
    else:
        if isinstance(node['r_subtree'], dict):
            return predict(node['r_subtree'], row)
        else:
            return node['r_subtree']


# In[ ]:




