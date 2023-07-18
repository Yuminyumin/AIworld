import pandas as pd
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# 나머지 코드 생략


import nltk
nltk.download('stopwords')

# Ignore noise warning
import warnings
warnings.filterwarnings("ignore")

MBTI_DS = pd.read_csv("c:\\Users\\신유민\\Desktop\\MBTI 500.csv")

nRow, nCol = MBTI_DS.shape
print(f'There are {nRow} rows and {nCol} columns')

types = np.unique(np.array(MBTI_DS['type']))
print("The Unique values 'type' of personality column", types)

total = MBTI_DS.groupby(['type']).count() * 50
print("The Total Posts for every Personality Type")
total

MBTI_DS_C = MBTI_DS.copy()

def var_row(row):
    l = []
    for i in row.split('|||'):
        l.append(len(i.split()))
    return np.var(l)

MBTI_DS_C['word_each_comment'] = MBTI_DS_C['posts'].apply(lambda x: len(x.split()) / 50)
MBTI_DS_C['variance_word_count'] = MBTI_DS_C['posts'].apply(lambda x: var_row(x))

MBTI_DS["length_posts"] = MBTI_DS["posts"].apply(len)

lemmatiser = WordNetLemmatizer()

useless_words = stopwords.words("english")

unique_type_list = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP','ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']
unique_type_list = [x.lower() for x in unique_type_list]

b_Pers = {'I':0, 'E':1, 'N':0, 'S':1, 'F':0, 'T':1, 'J':0, 'P':1}

def translate_personality(personality):
    return [b_Pers[l] for l in personality]

def translate_back(personality):
    s = ""
    for i, l in enumerate(personality):
        s += ['I', 'E'][l]
        if i % 2 != 0:
            s += " | "
    return s.rstrip(" | ")

list_personality_bin = np.array([translate_personality(p) for p in MBTI_DS['type']])
print("Binarize MBTI list: \n%s" % list_personality_bin)

nltk.download('wordnet')

STOPWORDS = set(stopwords.words('english'))

def pre_process_text(MBTI_DS, remove_stop_words=True, remove_mbti_profiles=True):
    list_personality = []
    list_posts = []
  
    for row in MBTI_DS.iterrows():
        posts = row[1].posts

        temp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', posts)

        temp = re.sub("[^a-zA-Z]", " ", temp)

        temp = re.sub(' +', ' ', temp).lower()

        temp = re.sub(r'([a-z])\1{2,}[\s|\w]*', '', temp)

        if remove_stop_words:
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ') if w not in useless_words])
        else:
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ')])
          
        if remove_mbti_profiles:
            for t in unique_type_list:
                temp = temp.replace(t, "")

        type_labelized = translate_personality(row[1].type)
        list_personality.append(type_labelized)
        list_posts.append(temp)

    list_posts = np.array(list_posts)
    list_personality = np.array(list_personality)
    return list_posts, list_personality

list_posts, list_personality  = pre_process_text(MBTI_DS, remove_stop_words=True, remove_mbti_profiles=True)

print("Example :")
print("\nPost before preprocessing:\n\n", MBTI_DS.posts[0])
print("\nPost after preprocessing:\n\n", list_posts[0])
print("\nMBTI before preprocessing:\n\n", MBTI_DS.type[0])
print("\nMBTI after preprocessing:\n\n", list_personality[0])

nRow, nCol = list_personality.shape
print(f'Number of posts = {nRow} and No. of Personalities = {nCol} ')

cntizer = CountVectorizer(analyzer="word", 
                          max_features=1000,  
                          max_df=0.7,
                          min_df=0.1) 

X_cnt = cntizer.fit_transform(list_posts)

feature_names = list(enumerate(cntizer.get_feature_names()))

tfizer = TfidfTransformer()
X_tfidf =  tfizer.fit_transform(X_cnt).toarray()
print(X_tfidf.shape)

personality_type = ["IE: Introversion (I) | Extroversion (E)", "NS: Intuition (N) | Sensing (S)", 
                    "FT: Feeling (F) | Thinking (T)", "JP: Judging (J) | Perceiving (P)"]

for l in range(len(personality_type)):
    print(personality_type[l])

print("For MBTI personality type: %s" % translate_back(list_personality[0,:]))
print("Y: Binarized MBTI 1st row: %s" % list_personality[0,:])

X = X_tfidf

for l in range(len(personality_type)):
    Y = list_personality[:,l]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=7)

    model = LogisticRegression() 
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("%s Accuracy: %.2f%%" % (personality_type[l], accuracy * 100.0))
    print("\n")

for l in range(len(personality_type)):
    Y = list_personality[:,l]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=7)

    model = XGBClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("%s Accuracy: %.2f%%" % (personality_type[l], accuracy * 100.0))
    print("\n")
