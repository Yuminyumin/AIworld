import numpy as np
import requests
import json
from enum import Enum

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

class MbtiAnalysis:
    def __init__(self, member_mbti_dict):
        self.mbti_dict = member_mbti_dict
        self.member = list(member_mbti_dict.keys())
        self._init_vibe()

    def _init_vibe(self):
        self.mbti_distance_matrix = np.zeros((len(self.member), len(self.member)))

        for i, mem1 in enumerate(self.member):
            for j, mem2 in enumerate(self.member):
                mbti1 = self.mbti_dict[mem1].value
                mbti2 = self.mbti_dict[mem2].value
                self.mbti_distance_matrix[i, j] = MBTI_GOODNESS[mbti1-1, mbti2-1]

    def print_vibe(self):
        for i, mem1 in enumerate(self.member):
            for j, mem2 in enumerate(self.member):
                print(f"{mem1}와 {mem2}의 궁합 점수: {self.mbti_distance_matrix[i, j]}")

    def get_vibe_scores(self):
        vibe_result = {}

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

response = requests.get("https://example.com/api/mbti")
data = json.loads(response.text)

member_mbti_dict = {}
for key, value in data.items():
    member_mbti_dict[key] = MBTI(value)

mbti_analysis = MbtiAnalysis(member_mbti_dict)
mbti_analysis.print_vibe()

vibe_scores = mbti_analysis.get_vibe_scores()
payload = json.dumps(vibe_scores)

headers = {
    'Content-Type': 'application/json'
}

response = requests.post("https://example.com/your-api", data=payload, headers=headers)

print("Status code:", response.status_code)
print("Response text:", response.text)
