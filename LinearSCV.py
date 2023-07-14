import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from tqdm.notebook import tqdm
from nltk.stem.snowball import SnowballStemmer
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('c:\\Users\\신유민\\Desktop\\MBTI 500.csv', encoding='utf-8')
train.head()
train.tail()
train
test = train.drop(['type'], axis=1)
test.head()

# 설명변수
X = train['posts']
# 예측변수
Y = train['type']

print(f"{len(train['type'].unique())}개")
train['type'].value_counts()

# plt.pie(train['type'])
train.isnull().sum()
train['posts'].nunique() == len(train['posts'])

print("train data : ", train.shape)
print("test data : ", test.shape)

# E, I 빈도수 확인
first = []
for i in range(len(train)):
    first.append(train['type'][i][0])
first = pd.DataFrame(first)
first[0].value_counts()

# N, S 빈도수 확인
second = []
for i in range(len(train)):
    second.append(train['type'][i][1])
second = pd.DataFrame(second)
second[0].value_counts()

# T, F 빈도수 확인
third = []
for i in range(len(train)):
    third.append(train['type'][i][2])
third = pd.DataFrame(third)
third[0].value_counts()

# P, J 빈도수 확인
fourth = []
for i in range(len(train)):
    fourth.append(train['type'][i][3])
fourth = pd.DataFrame(fourth)
fourth[0].value_counts()

X = train['posts'] # data features
Y = train['type'] # labels
X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.2, random_state=1) # test size = 0.3도 해보기

# 벡터화
tfidf = TfidfVectorizer()

# 훈련 데이터 벡터화
X_train_tfidf = tfidf.fit_transform(X_train)

clf = LinearSVC()
# 정확도 기준 설정
cv = GridSearchCV(clf, {'C': [0.35, 0.4, 0.45]}, scoring = "accuracy")
cv.fit(X_train_tfidf, y_train)

C = cv.best_params_['C']
print("최적의 파라미터 C: ", C)

text_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC(C=C))])
text_clf.fit(X_train, y_train)

# valid 데이터의 mbti 예측
pred = text_clf.predict(X_valid)

# valid data에서의 정확도
accuracy = accuracy_score(pred, y_valid)
print(f'Accuracy: {accuracy}')
