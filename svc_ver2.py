import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import *
from nltk.stem.snowball import SnowballStemmer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import pickle
from sklearn.metrics import recall_score

# 데이터 불러오기
train = pd.read_csv('c:/Users/user/Desktop/mbtidata/mbti_total.csv', encoding='latin1')
test = train.drop(['type'], axis=1)

# 설명변수(X)와 예측변수(Y) 설정
X = train['posts']
Y = train['type']

# 유니크한 MBTI 유형 개수 확인
print(f"{len(train['type'].unique())}개")

# 각 MBTI 유형별 데이터 개수 확인
class_counts = train['type'].value_counts()
print(class_counts)

# 결측값을 확인합니다.
print("결측값 수 (train.posts):", train.posts.isnull().sum())
print("결측값 수 (train.type):", train.type.isnull().sum())

# 결측치 제거
train.dropna(subset=['type'], inplace=True)
train.dropna(subset=['posts'], inplace=True)

# 결측값을 재확인합니다.
print("결측값 수 (train.posts):", train.posts.isnull().sum())
print("결측값 수 (train.type):", train.type.isnull().sum())

# 중복되지 않은 게시글 개수와 전체 게시글 수 비교
train['posts'].nunique() == len(train['posts'])

# 훈련 데이터와 테스트 데이터 형태 확인
print("train data : ", train.shape)
print("test data : ", test.shape)

# # NLTK 불용어 처리를 위해 불용어 리스트 다운로드
# nltk.download('stopwords')
# nltk.download('punkt')

# NLTK에서 사용할 SnowballStemmer 로드
s_stemmer = SnowballStemmer(language='english')

# 불용어 제거 함수
def removeStopwords(s):
    if isinstance(s, str):  # 문자열인 경우에만 처리
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(s)
        new_words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(new_words)
    else:
        return s  # NaN 또는 비문자열 값인 경우 그대로 반환

# 어간 추출 함수
def replaceStemwords(s):
    if isinstance(s, str):  # 문자열인 경우에만 처리
        words = word_tokenize(s)
        new_words = [s_stemmer.stem(word) for word in words]
        return ' '.join(new_words)
    else:
        return s  # NaN 또는 비문자열 값인 경우 그대로 반환

# 진행 상황 표시를 위한 tqdm 설정
tqdm.pandas()

# 불용어 제거 및 어간 추출 적용
train['posts'] = train['posts'].progress_apply(removeStopwords)
train['posts'] = train['posts'].progress_apply(replaceStemwords)
train.posts = train.posts.progress_apply(lambda x: removeStopwords(replaceStemwords(x)))

# 테스트 데이터에도 불용어 제거 및 어간 추출 적용
test.posts = test.posts.progress_apply(lambda x: removeStopwords(replaceStemwords(x)))

# 훈련 데이터의 설명변수(X)와 예측변수(Y) 설정
X_processed = train.posts.fillna('', inplace=False).replace({r'\s+$': '', r'^\s+': ''}, regex=True)  # 빈 문자열로 NaN 값을 대체하고 불필요한 공백 제거
Y = train.type

# 훈련 데이터와 검증 데이터 분리
X_train, X_valid, Y_train, Y_valid = train_test_split(X_processed, Y, test_size=0.2, random_state=1)

# 선형 SVM 모델을 사용한 텍스트 분류를 위한 Pipeline 생성 및 모델 훈련
text_clf = Pipeline([('tfidf',TfidfVectorizer()),('clf',LinearSVC(C=0.3))])
text_clf.fit(X_train, Y_train)

# 학습된 모델을 pickle 파일로 저장
save_path = 'c:/Users/user/Desktop/mbtidata/MBTIgram.pkl'
with open(save_path, 'wb') as model_file:
    pickle.dump(text_clf, model_file)

# 저장된 모델을 다시 로드하여 예측 수행
with open(save_path, 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# valid 데이터의 mbti 예측
pred = loaded_model.predict(X_valid)
print("pred",pred)

# valid data에서의 정확도
accuracy = accuracy_score(pred, Y_valid)
print("accuracy",accuracy)

# 평균 recall 값 출력 (예: 'micro', 'macro', 'weighted')
average_recall = recall_score(Y_valid, pred, average='macro')
print("Average Recall:", average_recall)

#클래스 개수 확인
class_counts = train['type'].value_counts()
print(class_counts)

# 각 클래스별 recall 값 출력
class_recall = recall_score(Y_valid, pred, average=None)
print("Recall by Class:", class_recall)

#클래스 불균형 시각화
class_counts.plot(kind='bar')
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.show()