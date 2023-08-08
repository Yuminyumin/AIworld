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

# 데이터 불러오기
data1 = pd.read_csv('c:/Users/user/Desktop/mbtidata/mbti.csv', encoding='utf-8')  # mbti500
data2 = pd.read_csv('c:/Users/user/Desktop/mbtidata/mbti_twitter.csv', encoding='utf-8') #mbti_twitter
data3 = pd.read_csv('c:/Users/user/Desktop/mbtidata/mbti_1.csv', encoding='utf-8') #mbti_1

# 데이터 병합
all_data = pd.concat([data1, data2, data3], ignore_index=True)

# 내용변수와 타입변수 설정
X = all_data['posts']
Y = all_data['type']

# 레이블 인코딩
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)

# 훈련 데이터와 검증 데이터 분리
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y_encoded, test_size=0.3, random_state=1)  # Y_encoded로 변경

# 훈련 데이터 전처리
X_train = X_train.apply(lambda x: str(x) if pd.notna(x) else '')  # NaN 값을 빈 문자열로 대체
X_train = X_train.dropna()

# 텍스트 전처리
max_words = 5000  # 사용할 최대 단어 개수
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_valid_sequences = tokenizer.texts_to_sequences(X_valid)

max_length = 200  # 시퀀스의 최대 길이
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

# 딥러닝 모델 구축 (LSTM) with Hyperparameter Tuning
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=max_words, output_dim=64, input_length=max_length),
    tf.keras.layers.LSTM(units = 64, dropout=0.3, recurrent_dropout=0.2), # 조정 가능한 하이퍼파라미터
    tf.keras.layers.Dense(32, activation='relu'), # 조정 가능한 하이퍼파라미터
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax') 
])

# 모델 컴파일
learning_rate = 0.001  # 수정 가능한 학습률 값
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Early Stopping 콜백 정의
early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

# 모델 훈련
model.fit(X_train_padded, Y_train, epochs=10, batch_size=64, validation_data=(X_valid_padded, Y_valid),callbacks=[early_stopping]) # 조정 가능한 하이퍼파라미터

# 검증 데이터에서의 예측 및 평가
pred_probs = model.predict(X_valid_padded)
pred_labels = np.argmax(pred_probs, axis=1)
accuracy = accuracy_score(pred_labels, Y_valid)
print("Accuracy:", accuracy)