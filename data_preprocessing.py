import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

file1 = "./data_latih/Agreeableness.csv"
file2 = "./data_latih/Conscientiousness.csv"
file3 = "./data_latih/Extraversion.csv"
file4 = "./data_latih/Neuroticsm.csv"
file5 = "./data_latih/Openness to New Experiences.csv"
bhs_ind = [y for y in range(51,102)]
dataset = pd.DataFrame()

#Read csv
train1 = pd.read_csv(file1,encoding="latin-1",header=None,skiprows=[0]+bhs_ind,names=['caption'])
train1['label'] = pd.Series('A',train1.index)
train2 = pd.read_csv(file2,encoding="latin-1",header=None,skiprows=[0]+bhs_ind,names=['caption'])
train2['label'] = pd.Series('C',train2.index)
train3 = pd.read_csv(file3,encoding="latin-1",header=None,skiprows=[0]+bhs_ind,names=['caption'])
train3['label'] = pd.Series('E',train3.index)
train4 = pd.read_csv(file4,encoding="latin-1",header=None,skiprows=[0]+bhs_ind,names=['caption'])
train4['label'] = pd.Series('N',train4.index)
train5 = pd.read_csv(file5,encoding="latin-1",header=None,skiprows=[0]+bhs_ind,names=['caption'])
train5['label'] = pd.Series('O',train5.index)

#append to dataset
dataset = dataset.append(train1)
dataset = dataset.append(train2,ignore_index=True)
dataset = dataset.append(train3,ignore_index=True)
dataset = dataset.append(train4,ignore_index=True)
dataset = dataset.append(train5,ignore_index=True)

#encoding labels from dataset
lbl_enc = preprocessing.LabelEncoder()
y = lbl_enc.fit_transform(dataset.label.values)

#split train and validation
xtrain,xtest,ytrain,ytest = train_test_split(dataset.caption.values,y,stratify=y,random_state=1337,test_size=0.1,shuffle=True)

# Always start with these features. They work (almost) everytime!
tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')

#count vect feat engineeering
ctv = CountVectorizer(analyzer='word', token_pattern=r'\w+'
        ngram_range=(1,3), stop_words = 'english')

# Fitting TF-IDF to both training and test sets (semi-supervised learning)
tfv.fit(list(xtrain) + list(xtest))
xtrain_tfv =  tfv.transform(xtrain) 
xtest_tfv = tfv.transform(xtest)

# Fitting CountVectorizer to both training and test set 
ctv.fit(list(xtrain) + list(xtest))
xtrain_ctv =  ctv.transform(xtrain) 
xtest_ctv = ctv.transform(xtest)

#logloss
def multiclass_logloss(actual,predicted,eps=1e-15):
    """Multi class version of logarithmic loss metric.
    :param actual: Array containing the actual target classes
    :param predicted: Matrix with class prediction, one probability per class
    """
    #convert 'actual' to a binary array if it's not already:
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0],predicted.shape[1]))
        for i,val in enumerate(actual):
            actual2[i,val]=1
        actual = actual2
    
    clip = np.clip(predicted,eps,1-eps)
    rows = actual.shape[0]
    vsota = np.sum(actual*np.log(clip))
    return -1.0 / rows * vsota

# Training with logistic regression using tfidf as feature engineering
clf = LogisticRegression(C=1.0)
clf.fit(xtrain_tfv,ytrain)

# fit the trained logreg with tfidf to test data
predictions = clf.predict_proba(xtest_tfv)

# Training with logistic regression using count as feature engineering
clf1 = LogisticRegression(C=1.0)
clf1.fit(xtrain_ctv,ytrain)

# fit the trained logreg with count to test data
predictions1 = clf.predict_proba(xtest_ctv)

print("logloss of logreg with tfidf: %0.3f " % multiclass_logloss(ytest,predictions))
print("logloss of logreg with count: %0.3f " % multiclass_logloss(ytest,predictions1))

