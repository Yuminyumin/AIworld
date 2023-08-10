import numpy as np
import requests
import json
from enum import Enum

# MBTI 유형을 나타내는 Enum 클래스 정의
class MBTI(Enum):
    INFP=0,
    ENFP=1,
    INFJ=2,
    ENFJ=3,
    INTJ=4,
    ENTJ=5,
    INTP=6,
    ENTP=7,
    ISFP=8,
    ESFP=9,
    ISTP=10,
    ESTP=11,
    ISFJ=12,
    ESFJ=13,
    ISTJ=14,
    ESTJ=15

# MBTI 유형 간 호환성 점수 행렬
MBTI_GOODNESS = np.matrix([
    [4, 4, 4, 5, 4, 5, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1],
    [4, 4, 5, 4, 5, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1],
    [4, 5, 4, 4, 4, 4, 4, 5, 1, 1, 1, 1, 1, 1, 1, 1],
    [5, 4, 4, 4, 4, 4, 4, 4, 5, 1, 1, 1, 1, 1, 1, 1],
    [4, 5, 4, 4, 4, 4, 4, 5, 3, 3, 3, 3, 2, 2, 2, 2],
    [5, 4, 4, 4, 4, 4, 5, 4, 3, 3, 3, 3, 3, 3, 3, 3],
    [4, 4, 4, 4, 4, 5, 4, 4, 3, 3, 3, 3, 2, 2, 2, 5],
    [4, 4, 5, 4, 5, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2],
    [1, 1, 1, 5, 3, 3, 3, 3, 2, 2, 2, 2, 3, 5, 3, 5],
    [1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2, 5, 3, 5, 3],
    [1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2, 3, 5, 3, 5],
    [1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2, 5, 3, 5, 3],
    [1, 1, 1, 1, 2, 3, 2, 2, 3, 5, 3, 5, 4, 4, 4, 4],
    [1, 1, 1, 1, 2, 3, 2, 2, 5, 3, 5, 3, 4, 4, 4, 4],
    [1, 1, 1, 1, 2, 3, 2, 2, 3, 5, 3, 5, 4, 4, 4, 4],
    [1, 1, 1, 1, 2, 3, 5, 2, 5, 3, 5, 3, 4, 4, 4, 4]
])

# MBTI 호환성 분석을 수행하는 클래스 정의 
class MbtiAnalysis:
    def __init__(self, member_mbti_dict):
        self.mbti_dict = member_mbti_dict  # 유저들의 이름과 MBTI 정보를 담은 딕셔너리
        self.member = list(member_mbti_dict.keys())  # 유저들의 이름 리스트
        self._init_vibe()  # mbti_distance_matrix를 초기화하는 메서드 호출

    def _init_vibe(self):
        # 멤버들 간의 호환성 점수를 저장할 행렬 초기화
        self.mbti_distance_matrix = np.zeros((len(self.member), len(self.member)))

        # 멤버들간의 호환성 계산 및 행렬에 저장
        for i, mem1 in enumerate(self.member):
            for j, mem2 in enumerate(self.member):
                mbti1 = self.mbti_dict[mem1].value
                mbti2 = self.mbti_dict[mem2].value
                self.mbti_distance_matrix[i, j] = MBTI_GOODNESS[mbti1-1, mbti2-1]

    def print_vibe(self):
        # 각 멤버 간의 호환성 점수 출력
        for i, mem1 in enumerate(self.member):
            for j, mem2 in enumerate(self.member):
                print(f"{mem1}와 {mem2}의 궁합 점수: {self.mbti_distance_matrix[i, j]}")

    def get_vibe_scores(self):
        vibe_result = {}  # 결과를 저장할 딕셔너리 초기화

        # 결과 딕셔너리에 각 멤버 간의 호환성 점수 저장
        for i, mem1 in enumerate(self.member):
            for j, mem2 in enumerate(self.member):
                key = f"{mem1}-{mem2}"
                vibe_value = self.mbti_distance_matrix[i, j]
                vibe_result[key] = {
                    "mbti1": self.mbti_dict[mem1].name,
                    "mbti2": self.mbti_dict[mem2].name,
                    "vibe": vibe_value
                }

        return vibe_result
    
# API에서 MBTI 정보를 가져옵니다. 
response = requests.get("https://example.com/api/mbti")
data = json.loads(response.text)

# 각 멤버의 MBTI 유형 정보를 딕셔너리 형태로 저장합니다. 
member_mbti_dict = {}
for key, value in data.items():
    member_mbti_dict[key] = MBTI(value)

# MbtiAnalysis 클래스 객체를 생성하여, 각 멤버별 분위기(vibe) 점수를 분석합니다. 
mbti_analysis = MbtiAnalysis(member_mbti_dict)
mbti_analysis.print_vibe()

# 분위기(vibe) 점수를 API 서버로 전송하기 위해, JSON 형태로 데이터를 준비합니다.
vibe_scores = mbti_analysis.get_vibe_scores()
payload = json.dumps(vibe_scores)

# HTTP POST 요청을 사용하여 데이터를 API 서버로 전송합니다. 
headers = {
    'Content-Type': 'application/json'
}

response = requests.post("https://example.com/your-api", data=payload, headers=headers)

# 응답 결과를 출력합니다.
print("Status code:", response.status_code)
print("Response text:", response.text)
