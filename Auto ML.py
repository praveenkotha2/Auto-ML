# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 19:43:42 2018

@author: PraveenKotha
"""
#Importing required packages
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.base import TransformerMixin
from sklearn.preprocessing import Imputer
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.linear_model import Ridge


#Importing data
data = pd.read_csv('/Users/PraveenKotha/Desktop/TrainBigmart.csv')

###
def setIndex(datai):
    for i in np.arange(0,1):
        if (datai[datai.columns[i]].nunique() == datai.shape[0]):
            datai = datai.set_index(datai.columns[i],drop = True)
            return datai
        else:
            return datai
        
data = setIndex(data)  

###
def ioSeparator(dataio):
    a = input("Enter the target output column")
    if a not in dataio.columns:
        print("There is no column present with that name, try again")
    else:   
        y = dataio[a]
        b = dataio.columns.drop(a)
        X = dataio[b]
        return X,y

X, y = ioSeparator(data)


def problemIdentifierSplit(X_pi,y_pi):
    n = y_pi.nunique()
    if (n==2 | n < 6):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y)
    else :
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = problemIdentifierSplit(X,y)


   
def typesep(datat):
    for i in np.arange(0,len(datat.columns)):
        if ((datat.loc[:,datat.columns[i]].nunique() <= 10) & (datat.loc[:,datat.columns[i]].dtype == 'O')):
            datat.loc[:,datat.columns[i]] = datat.loc[:,datat.columns[i]].astype('category').copy()
        elif (datat.loc[:,datat.columns[0]].dtype == 'int64'):
            datat.loc[:,datat.columns[i]] = datat.loc[:,datat.columns[i]].astype('int64').copy() 
        elif (datat.loc[:,datat.columns[0]].dtype == 'float64'):
            datat.loc[:,datat.columns[i]] = datat.loc[:,datat.columns[i]].astype('float64').copy()
        
        
    datan = datat.select_dtypes(include = ['float64','int64'])
    datac = datat.select_dtypes(include = ['category']) 
    datat = datat.select_dtypes(include = ['object'], exclude = ['float64','int64','category'])
       
    return datan,datac,datat               

X_n, X_c, X_t = typesep(X_train)   


def imputerNum(dataimpn):
    dataimpn = dataimpn.apply(lambda x : x.fillna(x.mean()))
    return dataimpn
      
X_n = imputerNum(X_n)            


def standardizeNum(datasn):
    datasn = datasn.apply(lambda x : (x - np.mean(x))/np.std(x))
    return datasn

X_n = standardizeNum(X_n) 


def imputerCat(dataimpc):
    dataimpc = dataimpc.apply(lambda x : x.fillna(x.mode()[0]))
    return dataimpc

X_c = imputerCat(X_c) 
X_t = imputerCat(X_t) 


def categoryOneEncode(trainc):
    trainc = pd.get_dummies(trainc)
    return trainc

X_c = categoryOneEncode(X_c) 

def textProcessing(datatp):
    datatp = datatp.apply(lambda x : '%s %s' %(x[0],x[1]), axis = 1)
    datatp = pd.DataFrame(datatp)
    datatp.columns = ['text']
    datatp['text'] = datatp['text'].astype(str)
    return datatp
    
X_t = textProcessing(X_t) 

from nltk import word_tokenize
X_t['text'] = X_t['text'].apply(word_tokenize)

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
X_text= vect.fit_transform(X_t)

def columnStack(train1,train2,train3):
    return pd.concat([train1,train2,train3],axis = 1)

X_total = columnStack(X_c,X_n,X_t)


def featureSelection(X_fs, y_fs):
    if(y_fs.nunique() == 2):
        clf = RandomForestClassifier(n_estimators = 100, n_jobs = -1)
        clf.fit(X_fs, y_fs)
        for feature in zip(X_fs.columns, clf.feature_importances_):
            print(feature)
        sfm = SelectFromModel(clf, threshold = 0.005)    
        sfm.fit(X_fs, y_fs)
        fs = []
        for feature_list_index in sfm.get_support(indices=True):
            fs.append(feature_list_index)
        X_imp = X_fs.iloc[:,fs] 
        return X_imp,y_fs
    else:
        clf = RandomForestRegressor(n_estimators = 100, n_jobs = -1)
        clf.fit(X_fs, y_fs)
        for feature in zip(X_fs.columns, clf.feature_importances_):
            print(feature)
        sfm = SelectFromModel(clf, threshold = 0.005)    
        sfm.fit(X_fs, y_fs)
        fs = []
        for feature_list_index in sfm.get_support(indices=True):
            fs.append(feature_list_index)
        X_imp = X_fs.iloc[:,fs] 
        return X_imp,y_fs
        
    
X_total, y_train = featureSelection(X_total, y_train)        


def compareModels1(X_m, y_m):
    if (y_m.nunique()==2) : 
        
        models = []
        models.append(('LR',LinearRegression(),
                   {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                   'penalty': ['l1','l2']}))
        models.append(('KNN',KNeighborsClassifier(),
                   {'n_neighbors': [2, 4, 8, 16],
                    'p': [2, 3]}))
        models.append(('NB',GaussianNB(),
                   {}))
        models.append(('SVM',SVC(),
                   {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                    'gamma': ['auto'],
                    'class_weight': ['balanced',None]}))
    
        results = []
        names = []
        params = []
        scoring = 'accuracy'
    
        for name,model,param_grid in models:
            cv = GridSearchCV(model, param_grid, cv = 5, scoring=scoring )
            cv.fit(X_m,y_m)
        
            results.append(cv.best_score_)
            params.append(cv.best_params_)
            names.append(name)
        return results,names,params
    else :
        models1 = []
        models1.append(('LiR',LinearRegression(),
                   {'fit_intercept': [True, False],
                   'normalize': [True, False]}))
        models1.append(('lasso',Lasso(),
                   {'alpha': [0.1, 1, 10],
                    'normalize': [True, False]}))
        models1.append(('ridge',Ridge(),
                   {'alpha': [0.01, 0.1, 1, 10, 100],
                    'fit_intercept': [True, False],
                   'normalize': [True, False]}))
        models1.append(('SVM',SVR(),
                   {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'epsilon':[0.1,0.2,0.5,0.3]
                    }))
    
        results1 = []
        names1 = []
        params1 = []
        scoring1 = 'r2'
    
        for name1,model1,param_grid1 in models1:
            cv1= GridSearchCV(model1, param_grid1, cv = 5, scoring=scoring1 )
            cv1.fit(X_m,y_m)
        
            results1.append(cv1.best_score_)
            params1.append(cv1.best_params_)
            names1.append(name1)
        return results1,names1,params1
        
result1, names1, params1 = compareModels1(X_total, y_train) 


