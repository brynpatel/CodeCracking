import csv
from collections import Counter
import re
import numpy as np
import itertools
import pandas as pd
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

alphabet = "abcdefghijklmnopqrstuvwxyz"
dict={}
rows = []
fdict = {}
ddict = {}
adict = {}
bdict = {}
b2dict = {}
a2dict = {}
letterBefore = ""
letterAfter = ""
letter2Before = ""
letter2After = ""
lettersBefore = []
lettersAfter = []
letters2Before = []
letters2After = []
positioning = []
freq2Following = {}
freqFollowing = {}
freq2Preceding = {}
freqPreceding = {}

with open ("plainText.txt", "r") as myfile:
    file=myfile.readlines()

for line in file:
    dict = {}
    for c in alphabet:
        dict[c] = 0
        freqPreceding[c + "Precede"] = 0
        freqFollowing[c + "Follow"]  = 0
        freq2Following[c + "2Follow"] = 0
        freq2Preceding[c + "2Preced"] = 0
    line = re.sub(r"[^a-zA-Z]","",line)
    line = line.lower()
    if len(line) == 0:
        pass
    frequency=Counter(line)
    dict.update(frequency)
    for j in dict:
        freq2Following = {}
        freqFollowing = {}
        freq2Preceding = {}
        freqPreceding = {}
        for c in alphabet:
            freqPreceding[c + "Precede"] = 0
            freqFollowing[c + "Follow"]  = 0
            freq2Following[c + "2Follow"] = 0
            freq2Preceding[c + "2Preced"] = 0
        cPreceding = {}
        cFollowing = {}
        c2Preceding = {}
        c2Following = {}
        letterBefore = 0
        letterAfter = 0
        fdict = {}
        Fdict = {}
        distribution = []
        distance = 0
        location = 0
        lettersBefore = []
        lettersAfter = []
        positioning = []
        location = -1
        if dict[j] != 0:
            fdict["frequency"] = dict[j]/len(line)
        else:
            fdict["frequency"] = 0
        while True:
            location = line.find(j, location+1)
            if location == -1:
                break
            else:
                positioning.append(location)
                distribution.append(location+1)
        if len(distribution) == 0:
            distance = 0
            positioning.append(-1)
        elif len(distribution) == 1:
            distance = len(line)-1
        else:
            distance = [distribution[i+1]-distribution[i] for i in range(len(distribution)-1)]
            distance = sum(distribution)/len(distribution)
        for instance in positioning:
            if instance == 0: # at first letter
                letterBefore = 0 # no letters before
                letterAfter = line[instance+1] + "Follow"
                lettersAfter.append(letterAfter)
            elif instance == len(line)-1: #at last letter
                letterBefore = line[instance-1]+"Precede"
                letterAfter = 0 # no letters after
                lettersBefore.append(letterBefore)
            elif instance == -1: # letter not found
                letterBefore = 0 # no letters before
                letterAfter = 0 # no letters after
            else:
                # in the middle somewhere letters before and after
                letterBefore = line[instance-1]+"Precede"
                letterAfter = line[instance+1] + "Follow"
                lettersBefore.append(letterBefore)
                lettersAfter.append(letterAfter)

            # check
            if instance == -1: # letter not found
                letter2Before = 0 # no letters before
                letter2After = 0 # no letters after
            elif instance-2 <= -1 and instance+2 >= len(line)-1: # middle letter in three letters in the line OR two letters in line
                letter2Before = 0 # no letter 2 before
                letter2After = 0 #no letter 2 after
                letters2After.append(letter2After)
            elif instance-2 <= -1: # first or second letter in line
                letter2Before = 0 # no letter 2 before
                letter2After = line[instance+2] + "2Follow"
            elif instance+2 >= len(line)-1: # penultimate or ultimate letter in line
                letter2Before = line[instance-2] + "2Preceding"
                letter2After = 0 # no letter 2 after
                letters2Before.append(letter2Before)
            else:
                letter2Before = line[instance-2] + "2Preceding"
                letter2After = line[instance+2] + "2Follow"
                letters2Before.append(letter2Before)
                letters2After.append(letter2After)
            if len(lettersAfter) == 0 or len(lettersBefore) == 0 or len(letters2Before) == 0 or len(letters2After) == 0:
                pass
            else:
                #pass
                cFollowing = Counter(lettersAfter)
                cPreceding = Counter(lettersBefore)
                c2Following = Counter(letters2After)
                c2Preceding = Counter(letters2Before)
                freqPreceding.update(cPreceding)
                freqFollowing.update(cFollowing)
                freq2Preceding.update(cPreceding)
                freq2Following.update(cFollowing)
        ddict["distanceBetween"] = distance
        Fdict.update(freqPreceding)
        Fdict.update(freqFollowing)
        Fdict.update(freq2Preceding)
        Fdict.update(freq2Following)
        # Fdict.update(adict)
        # Fdict.update(bdict)
        # Fdict.update(a2dict)
        # Fdict.update(b2dict)
        Fdict.update(ddict)
        Fdict.update(fdict)
        Fdict["letter"] = j
        rows.append(Fdict)
    #     print(freqPreceding)
    # break

dataset = pd.DataFrame(rows)
#
array = dataset.values
X = array[:,0:106]
y = array[:,106]
print(y)
print(dataset.head(20))

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1    )

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
