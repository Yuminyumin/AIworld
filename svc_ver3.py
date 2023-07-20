import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from nltk.stem.snowball import SnowballStemmer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm

# 데이터 불러오기
train = pd.read_csv('c:\\Users\\신유민\\Desktop\\MBTI 500.csv', encoding='utf-8')
test = train.drop(['type'], axis=1)

# 설명변수
X = train['posts']
# 예측변수
Y = train['type']

print(f"{len(train['type'].unique())}개")
print(train['type'].value_counts())
print(train.isnull().sum())
print(train['posts'].nunique() == len(train['posts']))
print("train data:", train.shape)
print("test data:", test.shape)

# NLTK 불용어 처리를 위해 불용어 리스트 다운로드
# nltk.download('stopwords')
# nltk.download('punkt')

# NLTK에서 사용할 SnowballStemmer 로드
s_stemmer = SnowballStemmer(language='english')
def removeStopwords(s):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(s)
    new_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(new_words)
# 어간 추출 함수
def replaceStemwords(s):
    words = word_tokenize(s)
    new_words = [s_stemmer.stem(word) for word in words]
    return ' '.join(new_words)

tqdm.pandas()
train['posts'] = train['posts'].progress_apply(removeStopwords)  # 불용어 처리 적용
train['posts'] = train['posts'].progress_apply(replaceStemwords)  # 어간 추출 적용

X = train.posts
Y = train.type

X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2, random_state=1)

# TfidfVectorizer를 사용하여 텍스트 데이터를 벡터화
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_valid_tfidf = vectorizer.transform(X_valid)

text_clf = Pipeline([('clf', LinearSVC(C=0.3))])
text_clf.fit(X_train_tfidf, Y_train)

# valid 데이터의 mbti 예측
pred = text_clf.predict(X_valid_tfidf)
print("pred", pred)

# valid data에서의 정확도
accuracy = accuracy_score(pred, Y_valid)
print("accuracy", accuracy)

# 테스트 데이터에도 전처리 적용
test['posts'] = test['posts'].progress_apply(removeStopwords) # 불용어 처리 적용
test['posts'] = test['posts'].progress_apply(replaceStemwords)

# 테스트 데이터를 벡터화
X_test_tfidf = vectorizer.transform(test['posts'])

# 테스트 데이터 예측
test_pred = text_clf.predict(X_test_tfidf)

# 예측 결과를 DataFrame으로 만듦
predictions = pd.DataFrame({'type': test_pred})

# 원래 테스트 데이터와 예측 결과를 합침
result = pd.concat([test, predictions], axis=1)
print(result.head())