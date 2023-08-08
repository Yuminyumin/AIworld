import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler


# 데이터 불러오기
data1 = pd.read_csv('c:/Users/user/Desktop/mbtidata/mbti.csv', encoding='utf-8')  # mbti500
data3 = pd.read_csv('c:/Users/user/Desktop/mbtidata/mbti_1.csv', encoding='utf-8') # mbti_1

# 데이터 병합
all_data = pd.concat([data1, data3], ignore_index=True)

# 내용변수와 타입변수 설정
X = all_data['posts']
Y = all_data['type']

# 레이블 인코딩
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)

# 훈련 데이터와 검증 데이터 분리
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y_encoded, test_size=0.3, random_state=1)

# 훈련 데이터와 검증 데이터의 NaN 값을 빈 문자열로 대체
X_train = X_train.apply(lambda x: str(x) if pd.notna(x) else '') 
X_valid = X_valid.apply(lambda x: str(x) if pd.notna(x) else '')

# 텍스트 전처리
max_words = 5000
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_valid_sequences = tokenizer.texts_to_sequences(X_valid)

max_length = 200
X_train_padded = pad_sequences(X_train_sequences, maxlen=max_length, padding='post', truncating='post')
X_valid_padded = pad_sequences(X_valid_sequences, maxlen=max_length, padding='post', truncating='post')

# 불용어 처리를 위한 준비
stop_words = set(stopwords.words('english'))

# 불용어 처리 함수 정의
def remove_stopwords(texts):
    return [[word for word in sentence if word not in stop_words] for sentence in texts]

# 텍스트 전처리 - 불용어 처리
X_train_padded = remove_stopwords(X_train_padded)
X_valid_padded = remove_stopwords(X_valid_padded)

# SMOTE를 적용하여 데이터 증강
smote = SMOTE(random_state=0)
X_train_resampled, Y_train_resampled = smote.fit_resample(X_train_padded, Y_train)

#오버샘플링을 적용한 훈련 데이터 생성
oversampler = RandomOverSampler(random_state=0)
X_train_oversampled, Y_train_oversampled = oversampler.fit_resample(X_train_padded, Y_train)

# # 병합한 데이터셋의 처음 5개 행 보기
# print("Head of the merged dataset:")
# print(all_data.head())

# # 병합한 데이터셋의 마지막 5개 행 보기
# print("Tail of the merged dataset:")
# print(all_data.tail())

# 딥러닝 모델 구축 (LSTM) with Hyperparameter Tuning
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=max_words, output_dim=64, input_length=max_length),
    tf.keras.layers.LSTM(units = 16, dropout=0.5, recurrent_dropout=0.2),                                                                       # 조정 가능한 하이퍼파라미터
    tf.keras.layers.Dense(16, activation='relu'),                                                                                               # 조정 가능한 하이퍼파라미터
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax') 
])

# 모델 컴파일
learning_rate = 0.01  # 수정 가능한 학습률 값
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 훈련 전에 데이터 변환
X_train_padded = np.array(X_train_padded)
X_valid_padded = np.array(X_valid_padded)
Y_train = np.array(Y_train)

# Early Stopping 콜백 정의
early_stopping = EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)

# 모델 훈련
model.fit(X_train_oversampled, Y_train_oversampled, epochs=7, batch_size=32, validation_data=(X_valid_padded, Y_valid), callbacks=[early_stopping]) # 조정 가능한 하이퍼파라미터

# 검증 데이터에서의 예측 및 평가
pred_probs = model.predict(X_valid_padded)
pred_labels = np.argmax(pred_probs, axis=1)
accuracy = accuracy_score(pred_labels, Y_valid)
print("Accuracy:", accuracy)

# True Positive (TP)와 False Negative (FN) 계산
TP = np.sum(np.logical_and(pred_labels == 1, Y_valid == 1))
FN = np.sum(np.logical_and(pred_labels == 0, Y_valid == 1))

# 재현율 계산
recall = TP / (TP + FN)
print("Recall:", recall)

# #클래스 개수 확인
# class_counts = all_data['type'].value_counts()
# print(class_counts)

# #클래스 불균형 시각화

# class_counts.plot(kind='bar')
# plt.xlabel('Class')
# plt.ylabel('Count')
# plt.title('Class Distribution')
# plt.show()