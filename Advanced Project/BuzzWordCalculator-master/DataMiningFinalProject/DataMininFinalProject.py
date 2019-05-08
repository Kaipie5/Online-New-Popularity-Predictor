#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 09:09:03 2018

@author: kaim
"""


import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation, metrics

from urllib.request import urlopen
from urllib.request import Request
import pickle
import gevent.monkey
import re
import html
from collections import Counter
import operator
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import export_graphviz
import graphviz
from sklearn import tree



data = pd.read_csv("OnlineNewsPopularity.csv")


urls = data["url"]
shares = data[" shares"]
titles = []

#######################WEB SCRAPE SECTION##################
for i in range(0,len(urls)):
    urlTest.append(urls[i])
    titles.append("Initial")


#gevent.monkey.patch_all()
#NLTK natural language processing kit

def titlesListBuilder(url):
    try:
        #print("NNNNNNNNNNNNNNNNNNNN", numPlace)
        print('Starting {}'.format(url))
        req = Request(url)
        response = urlopen(req)
        page = str(response.read())
        #print(page)
        match = re.search('<title>(.*?)</title>', page)
        title = match.group(1) if match else "ERROR"
        fixedTitle = html.unescape(title)
        #print(fixedTitle)
#        titleFix1 = str(title).replace("<title>", "")
#        titleFix2 = str(titleFix1).replace("</title>", "")
#        print(titleFix2)
        titles[numPlace] = fixedTitle
        print(fixedTitle)
        print(numPlace, "DONE")
    except:
        titles[numPlace] = "Error"
    
jobs = [gevent.spawn(titlesListBuilder, urlTest[i], i) for i in range(0, len(urlTest))]

gevent.wait(jobs)

#WEBSITE SCRAPER
for url in urls:
    try:
        #print('Starting {}'.format(url))
        req = Request(url)
        response = urlopen(req)
        page = str(response.read())
        #print(page)
        match = re.search('<title>(.*?)</title>', page)
        title = match.group(1) if match else "ERROR"
        fixedTitle = html.unescape(title)
        #print(fixedTitle)
    #        titleFix1 = str(title).replace("<title>", "")
    #        titleFix2 = str(titleFix1).replace("</title>", "")
    #        print(titleFix2)
        #titles[numPlace] = fixedTitle
        print(fixedTitle)
        #print(numPlace, "DONE")
        titles.append(fixedTitle)
    except:
        titles.append("Error")
        
print("ALL DONE")


print("TITLES", titles)

with open("titles", 'wb') as f:
    pickle.dump(titles, f)
####################WEB SCRAPE DONE

##READS TITLES FROM PREPROCESSED TITLES
with open("titles", 'rb') as f:
    readTitles = pickle.load(f)

###########################Organize Weight of Words in terms of shares and sort the dictionary    
titleDict = {}
k = 0
for title in readTitles:
    titleDict[title] = shares[k]
    k = k + 1

finalDict = {}
for title in titleDict:
    split = title.split()
    for word in split:
        finalDict[word] = [0, 0, 0, 0, 0, 0, 0, 0]
        

i = 0
for title in titleDict:
    wordDict = title.split()
    for word in wordDict:
        finalDict[word][0] = finalDict[word][0] + 1
        finalDict[word][1] = finalDict[word][1] + titleDict[title]
        if data[' data_channel_is_lifestyle'][i] == 1:
            finalDict[word][2] = finalDict[word][2] + titleDict[title]
        if data[' data_channel_is_entertainment'][i] == 1:
            finalDict[word][3] = finalDict[word][3] + titleDict[title]
        if data[' data_channel_is_bus'][i] == 1:
            finalDict[word][4] = finalDict[word][4] + titleDict[title]
        if data[' data_channel_is_socmed'][i] == 1:
            finalDict[word][5] = finalDict[word][5] + titleDict[title]
        if data[' data_channel_is_tech'][i] == 1:
            finalDict[word][6] = finalDict[word][6] + titleDict[title]
        if data[' data_channel_is_world'][i] == 1:
            finalDict[word][7] = finalDict[word][7] + titleDict[title]
    i = i + 1

def setup(index):
    fixedDict = {}
    for val in finalDict:
        fixedDict[val] = finalDict[val][index]/finalDict[val][0]
    #Sort
    fixedDict = sorted(fixedDict.items(), key=operator.itemgetter(1))
    fixedDict.reverse()
    #Analysis
    indexString = "Index" + str(index)
    with open(str(indexString), 'wb') as f:
        pickle.dump(fixedDict, f)

for x in range(1, 8):
    setup(x)  
    
foo = ["", "all", "lifestyle", "entertainment", "business", "medical", "tech", "world"]

def analyze(index):
    indexString = "Index" + str(index)
    with open(indexString, 'rb') as f:
        analysisDict = pickle.load(f)
    print()
    print()
    print()
    print()
    print("******************* ", foo[index])
    for i in range(0, 20):
        print(analysisDict[i])
    plotList = []
    for i in range(0, 800):
        plotList.append(analysisDict[i][1])
    plt.plot(plotList)
    
    
#Sum of frequency of all words in title 
    
for g in range(1, 8):
    analyze(g)

def newColumnSetUp(index):
    #Title: A B C
    #F(A) = shares of A
    #Max = top word in All
    #MAX(F(A)/F(MAX), F(B)/F(MAX), F(C)/F(MAX))  
    #Do this for each channell
    #Put all of these in as new columns in my data
    with open("titles", 'rb') as f:
        readTitles = pickle.load(f)
        
    indexString = "Index" + str(index)
    with open(indexString, 'rb') as f:
        analysisDict = pickle.load(f)
    maxShareWord = analysisDict[0][1]
    newColumn = []
    searchDict = []
    for i in range(0, len(analysisDict)):
        searchDict.append(analysisDict[i][0])
    k = 0
    for title in readTitles:
        if index == 1 or (index == 2 and data[' data_channel_is_lifestyle'][k]) or (index == 3 and data[' data_channel_is_entertainment'][k]) or (index == 4 and data[' data_channel_is_bus'][k]) or (index == 5 and data[' data_channel_is_socmed'][k]) or (index == 6 and data[' data_channel_is_tech'][k]) or (index == 7 and data[' data_channel_is_world'][k]):
            titleShares = data[' shares'][k]
            split = title.split()
            calculatedVal = 0.0
            for word in split:
                indexOfWord = searchDict.index(word)
                if calculatedVal < analysisDict[indexOfWord][1]/maxShareWord - titleShares/maxShareWord:
                    calculatedVal = analysisDict[indexOfWord][1]/maxShareWord - titleShares/maxShareWord
            newColumn.append(calculatedVal) 
        else:
            newColumn.append(0)
        k = k + 1
    columnName = foo[index] + " word prediction"
    data[columnName] = newColumn
    
    with open("FULLCSV", 'wb') as f:
        pickle.dump(data, f)

for g in range(1,8):
    newColumnSetUp(g)

#newColumnSetUp(2)


    
with open("FULLCSV", 'rb') as f:
    newData = pickle.load(f)




#print(newData)
#print(data)


##########################Words weighted and put in Dictionary File FINISHED####################
    

#########Size Of Word Analysis##########


wordLengthAnalysis = []
for v in range(0, 40):
    wordLengthAnalysis.append(0)

#print(wordLengthAnalysis)
    
for word in analysisDict:
    wordLengthAnalysis[len(word[0])] = wordLengthAnalysis[len(word)] + word[1]
    
removedOutliers = []
for i in range(0, len(wordLengthAnalysis)):
    if wordLengthAnalysis[i] > 1200000:
        removedOutliers.append(wordLengthAnalysis[i])
        
print(removedOutliers)
print("WORD COUNT ANALYSIS")
plt.plot(removedOutliers)


plt.hist()

#########Size Of Word Analysis FINISHED#########

########MODEL TESTING###########

popular = data[' shares'] >= 1400
unpopular = data[' shares'] < 1400

data.loc[popular,' shares'] = 1
data.loc[unpopular,' shares'] = 0

#split original dataset into 60% training and 40% testing
features=list(data.columns[2:60])
x_train, x_test, y_train, y_test = cross_validation.train_test_split(data[features], data[' shares'], test_size=0.4, random_state=0)

x_rest, x_test_part, y_rest, y_test_part= cross_validation.train_test_split(x_test, y_test,
                                                                            test_size=0.7, random_state=0)
print("DecisionTree on Original Data Set")
decisionTree = DecisionTreeClassifier(min_samples_split=20,random_state=99)
clf_dt=decisionTree.fit(x_train,y_train)
score_dt=clf_dt.score(x_test_part,y_test_part)
print("Acurracy: ", score_dt)

print("RandomForest on Original Data Set")
rf = RandomForestClassifier(n_estimators=100,n_jobs=-1)
clf_rf = rf.fit(x_train,y_train)
score_rf=clf_rf.score(x_test_part,y_test_part)
print("Acurracy: ", score_rf)

print("NaiveBayes on Original Data Set")
nb = BernoulliNB()
clf_nb=nb.fit(x_train,y_train)
score_nb=clf_nb.score(x_test_part,y_test_part)
print("Acurracy: ", score_nb)


print()
print()

popular = newData[' shares'] >= 1400
unpopular = newData[' shares'] < 1400

newData.loc[popular,' shares'] = 1
newData.loc[unpopular,' shares'] = 0

indices = []
for i in range(2,68):
    if i != 60:
        indices.append(i)


features=list(newData.columns[indices])
x_train, x_test, y_train, y_test = cross_validation.train_test_split(newData[features], newData[' shares'], test_size=0.4, random_state=0)

x_rest, x_test_part, y_rest, y_test_part= cross_validation.train_test_split(x_test, y_test,
                                                                            test_size=0.7, random_state=0)

print("DecisionTree on New Data set")
decisionTree = DecisionTreeClassifier(min_samples_split=20,random_state=99)
clf_dt=decisionTree.fit(x_train,y_train)
score_dt=clf_dt.score(x_test_part,y_test_part)
print("Acurracy: ", score_dt)

print("RandomForest on New Data set")
rf = RandomForestClassifier(n_estimators=100,n_jobs=-1)
clf_rf = rf.fit(x_train,y_train)
score_rf=clf_rf.score(x_test_part,y_test_part)
print("Acurracy: ", score_rf)

print("NaiveBayes on New Data set")
nb = BernoulliNB()
clf_nb=nb.fit(x_train,y_train)
score_nb=clf_nb.score(x_test_part,y_test_part)
predict_nb = clf_nb.predict_proba(x_test_part)
print("Acurracy: ", score_nb)

##########MODEL TESTING FINISHED############


