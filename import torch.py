import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from sklearn.model_selection import GridSearchCV

# ---- Preprocessing
# CSV 파일 로드
data = pd.read_csv(r'c:\Users\신유민\Desktop\MBTI 500.csv', encoding='utf-8')
s_data = data.sample(frac=1)  # 이건 샘플링하려고 shuffle
s_data_s = s_data[1000:10000]

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
num_class = len(label_encoder.classes_)

# 하이퍼파라미터 그리드 설정
param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [3, 7],
    'learning_rate': [0.1, 0.01]
}

# XGBoost 모델 초기화
model = xgb.XGBClassifier(objective='multi:softmax', num_class=num_class, random_state=42)

# GridSearchCV를 사용하여 하이퍼파라미터 튜닝
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=2, scoring='accuracy')
grid_search.fit(train_sparse_tfidf_matrix, y_train)

# 최적의 하이퍼파라미터 조합 출력
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# 최적의 모델로 재학습
best_model = grid_search.best_estimator_
best_model.fit(train_sparse_tfidf_matrix, y_train)

# 예측
preds = best_model.predict(test_sparse_tfidf_matrix)

# 예측 결과 출력
for idx, prediction in enumerate(preds):
    predicted_type = label_encoder.inverse_transform([prediction])[0]
    print(f"Sample {idx+1}: {predicted_type}")

# 정확도 계산
accuracy = accuracy_score(y_test, preds)
print(f'Accuracy: {accuracy}')
