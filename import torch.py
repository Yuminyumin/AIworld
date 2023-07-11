import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# CSV 파일 로드
data = pd.read_csv('c:\\Users\\신유민\\Desktop\\MBTI 500.csv')

# 레이블 인코딩
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['type'])

# TF-IDF 벡터화 객체 생성
tfidf_vectorizer = TfidfVectorizer()

# 텍스트 데이터에 TF-IDF 적용
tfidf_matrix = tfidf_vectorizer.fit_transform(data['posts'])

# 희소 행렬로 변환
sparse_tfidf_matrix = csr_matrix(tfidf_matrix)

# XGBoost 학습용 데이터 생성
dtrain = xgb.DMatrix(sparse_tfidf_matrix, label=data['label'])

# XGBoost 모델 초기화
model = xgb.XGBClassifier()

# 모델 학습
print("모델 학습 시작")
num_epochs = 100
for epoch in range(num_epochs):
    model.fit(sparse_tfidf_matrix, data['label'], verbose=True)
    progress = (epoch + 1) / num_epochs * 100
    print(f"진행률: {progress:.2f}%")
print("모델 학습 완료")

# 학습 완료 후 모델을 사용하여 예측 수행
print("예측 시작")
predictions = model.predict(sparse_tfidf_matrix, output_margin=True)
print("예측 완료")

# 예측 결과 출력
print(predictions)