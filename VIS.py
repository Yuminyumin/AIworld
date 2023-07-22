import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# 저장된 모델 불러오기
with open('c:/Users/신유민/Desktop/saved_model.pkl', 'rb') as file:
    text_clf = pickle.load(file)

train = pd.read_csv('c:/Users/신유민/Desktop/MBTI 500.csv', encoding='utf-8')
test = train.drop(['type'], axis=1)

# test data 전처리: 어간 추출 및 불용어 제거
s_stemmer = SnowballStemmer(language='english')

def removeStopwords(s):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(s)
    new_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(new_words)

def replaceStemwords(s):
    words = word_tokenize(s)
    new_words = [s_stemmer.stem(word) for word in words]
    return ' '.join(new_words)

# test 데이터에 전처리 적용
test.posts = test.posts.apply(lambda x: removeStopwords(replaceStemwords(x)))

# 예측 수행
X_valid = test['posts']
pred = text_clf.predict(X_valid)
print("예측값:", pred)

# 예측값 시각화
plt.figure(figsize=(10, 6))
plt.bar(range(len(pred)), pred, color='blue')
plt.title('예측값')
plt.xlabel('샘플 번호')
plt.ylabel('예측된 MBTI 유형')
plt.show()
