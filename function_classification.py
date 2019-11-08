#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np


# In[5]:


def test_split_c(index, value, x,y):
    left,left_y, right, right_y = list(), list(),list(), list() #инициализация left,left_y, right, right_y
    for i,row in enumerate(x): #  перебор по строкам
        if row[index] < value: # если в строке значение по index  меньше value
            left.append(row)     # Добавляем строку в левое поддерево
            left_y.append(y[i])  # Добавляем метки в левое поддерево
        else:
            right.append(row) # Добавляем строку в правое поддерево
            right_y.append(y[i])# Добавляем метки в правое поддерево
    return left,left_y, right,right_y


# In[6]:


def gini_index(left,left_y, right,right_y , class_values):
    Xm = len(left)+len(right) #Вычесляем общее количество эдементов в левом и правом поддереве
    gini = 0.0 # началбная оценка Gini
    
    for i,group in enumerate([left,right]):
        size_LR = len(group)
        
        if size_LR == 0:
            continue
        Q_score = 0.0
        for k in class_values:
            if i==0:
                Pk = [row for row in left_y].count(k) / size_LR
                c = size_LR/Xm
            else:
                Pk = [row for row in right_y].count(k) / size_LR
                c = size_LR/Xm
                
            gini += c*(Pk*(1 - Pk))
        Q_score += gini  
    return Q_score


# In[7]:


def entropy(left,left_y, right,right_y , class_values):
    Xm = len(left)+len(right) #Вычесляем общее количество эдементов в левом и правом поддереве
    
    ent = 0.0 # началбная оценка entropy
    for i,subtree in enumerate([left,right]):
        size_LR = len(subtree)

        if size_LR == 0:
            continue
        Q_score = 0.0
        for k in class_values:
            if i==0:
                Pk = [row for row in left_y].count(k) / size_LR
                c = size_LR/Xm
            else:
                Pk = [row for row in right_y].count(k) / size_LR
                c = size_LR/Xm
            ent += -(Pk*np.log1p(Pk))*c
        Q_score += ent   
        
    return Q_score


# In[8]:


def get_split_c(x,y,criterion):
    
    class_values = list(set(y)) # Все возможные классы
    score_c = 0 #Начальная оценка
    b_score = 100
    
    for index in range(len(x[0])): #Перебор по индексам строки
        for row in x:#Перебор по строкам

            
            left,left_y, right,right_y = test_split_c(index, row[index], x,y) # вызов функции разбиения с параметрами [индекс, условие разбиения, набор данных x и y]; 
                                                      # возвращает left_row и right_row
            if criterion =='Gini':    
                score_c = gini_index(left,left_y, right,right_y , class_values)
                
            elif criterion =='entropy':
                score_c = entropy(left,left_y, right,right_y , class_values)
                
            if score_c < b_score:
                b_score = score_c
                node = {'index':index,'values':row[index],'l_subtree':left,'l_subtree_y':left_y,'r_subtree':right,'r_subtree_y':right_y}        
    return node


# In[9]:


def to_terminal_c(left_y=[], right_y=[]):
    outcomes = left_y+right_y
    return max(set(outcomes), key=outcomes.count)


# In[10]:


# Create child splits for a node or make terminal

def split_c(node,max_depth, min_samples, depth,criterion):
    left, right = node['l_subtree'], node['r_subtree']
    left_y, right_y = node['l_subtree_y'], node['r_subtree_y'] 

    #Проверка на пустые поддеревья
    if not left or not right:
        node['l_subtree'] = node['r_subtree'] = to_terminal_c(left_y , right_y)
        return
    
    
    # Проверка на максимальную глубину
    if depth >= max_depth:
        #Если юольше то расчитываем значени я в листях
        node['l_subtree'], node['r_subtree'] = to_terminal_c(left_y), to_terminal_c(right_y)
        return
    
    # Проверка на минимальное количество примеров в листе левого поддерева
    if len(left) <= min_samples:
        #Если меньше или равно, то расчитываем значени я в листях
        node['l_subtree'] = to_terminal_c(left_y,right_y)
    else:
        #Иначе продплжаем расщеплять
        node['l_subtree'] = get_split_c(left,left_y,criterion)
        split_c(node['l_subtree'],max_depth, min_samples, depth+1,criterion)
        
        
    # Проверка на минимальное количество примеров в листе правого поддерева
    if len(right) <= min_samples:
        node['r_subtree'] = to_terminal_c(left_y,right_y)
    else:
        node['r_subtree'] = get_split_c(right,right_y,criterion)
        split_c(node['r_subtree'], max_depth, min_samples, depth+1,criterion)


# In[11]:


def predict_c(node, row): #Функция для предсказания значений
    tf = row[node['index']] < node['values']
    if tf:
        if isinstance(node['l_subtree'], dict):
            return predict_c(node['l_subtree'], row)
        else:
            return node['l_subtree']
    else:
        if isinstance(node['r_subtree'], dict):
            return predict_c(node['r_subtree'], row)
        else:
            return node['r_subtree']

