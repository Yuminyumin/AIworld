import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import xgboost as xgb

# ---- Preprocessing
# CSV 파일 로드
data = pd.read_csv(r'c:\Users\신유민\Desktop\MBTI 500.csv', encoding='utf-8')
s_data = data.sample(frac=1) # 이건 샘플링하려고 shuffle

s_data_s = s_data[1000:10000] # 샘플링 용이고 다 쓰려면 s_data_s = data로 하면 돼

# 텍스트 데이터와 레이블 분리
X = s_data_s['posts']
y = s_data_s['type']

# 레이블 인코딩
label_encoder = LabelEncoder()
label_y = label_encoder.fit_transform(y)

# 학습 데이터와 테스트 데이터로 분할
X_train, X_test, y_train, y_test = train_test_split(X, label_y, test_size=0.3, random_state=42)

# TF-IDF 벡터화 객체 생성
tfidf_vectorizer = TfidfVectorizer()

# 텍스트 데이터에 TF-IDF 적용
train_tfidf_matrix = tfidf_vectorizer.fit_transform(X_train)
test_tfidf_matrix = tfidf_vectorizer.transform(X_test)

# 희소 행렬로 변환
train_sparse_tfidf_matrix = csr_matrix(train_tfidf_matrix)
test_sparse_tfidf_matrix = csr_matrix(test_tfidf_matrix)

# XGBoost 학습용 데이터 생성
dtrain = xgb.DMatrix(train_sparse_tfidf_matrix, label=y_train, enable_categorical=True)
dtest = xgb.DMatrix(test_sparse_tfidf_matrix, label=y_test, enable_categorical=True)

# ----- Model
num_class = len(pd.Categorical(y).categories)

# XGBoost 모델 초기화
params = {"objective":'multi:softmax', 
                            "num_class":num_class, 
                            "seed":42}
n = 100 # 이건 맘대로 boost round 정해주면 되는 변수

evals  = [(dtrain, "train"), (dtest, "validation")]
  
# Train
model = xgb.train(params=params,
                    dtrain=dtrain,
                    num_boost_round=n,
                    evals = evals)

# Predict
preds = model.predict(dtest)
print(preds)

# 학습 데이터에서 유형과 해당 인덱스 간의 매핑 생성
type_mapping = {}
for idx, mbti_type in enumerate(label_encoder.classes_):
    type_mapping[idx] = mbti_type

# 예측 결과를 MBTI 유형으로 변환
predicted_types = [type_mapping[prediction] for prediction in preds]

# 예측 결과 출력
for idx, predicted_type in enumerate(predicted_types):
    print(f"Sample {idx+1}: {predicted_type}")

accuracy = accuracy_score(y_test, predicted_types)
print(f'Accuracy: {accuracy}')