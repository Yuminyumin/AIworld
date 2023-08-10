import numpy as np

# MBTI 유형을 나타내는 Enum 클래스 정의
MBTI = {'INFP': 0,
        'ENFP': 1,
        'INFJ': 2,
        'ENFJ': 3,
        'INTJ': 4,
        'ENTJ': 5,
        'INTP': 6,
        'ENTP': 7,
        'ISFP': 8,
        'ESFP': 9,
        'ISTP': 10,
        'ESTP': 11,
        'ISFJ': 12,
        'ESFJ': 13,
        'ISTJ': 14,
        'ESTJ': 15}

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
        self.result_dict = self._init_vibe()

    def _init_vibe(self):
        # 멤버들 간의 호환성 점수를 저장할 행렬 초기화
        self.mbti_distance_matrix = np.zeros((len(self.member), len(self.member)))

        vibe_result = {}

        # 멤버들간의 호환성 계산 및 행렬에 저장
        for i in range(len(self.member)):
            mbti1 = self.mbti_dict[self.member[i]]
            for j in range(i+1, len(self.member)):
                mbti2 = self.mbti_dict[self.member[j]]
                self.mbti_distance_matrix[i, j] = MBTI_GOODNESS[mbti1 - 1, mbti2 - 1]

                couple = f"{self.member[i]}-{self.member[j]}"
                vibe_value = self.mbti_distance_matrix[i, j]
                vibe_result[couple] = {
                    "mbti1": self.mbti_dict[self.member[i]],
                    "mbti2": self.mbti_dict[self.member[j]],
                    "vibe": vibe_value
                }

                if j != i:
                    self.mbti_distance_matrix[j, i] = self.mbti_distance_matrix[i, j]
        return vibe_result

    def print_vibe(self):
        # 각 멤버 간의 호환성 점수 출력
        for i in range(len(self.member)):
            mem1 = self.member[i]
            for j in range(i+1, len(self.member)):
                mem2 = self.member[j]
                print(f"{mem1}와 {mem2}의 궁합 점수: {self.mbti_distance_matrix[i, j]}")

    def get_vibe_scores(self):
        vibe_result = {}  # 결과를 저장할 딕셔너리 초기화

        # 결과 딕셔너리에 각 멤버 간의 호환성 점수 저장
        for i, mem1 in enumerate(self.member):
            for j, mem2 in enumerate(self.member):
                key = f"{mem1}-{mem2}"
                vibe_value = self.mbti_distance_matrix[i, j]
                vibe_result[key] = {
                    "mbti1": self.mbti_dict[mem1],
                    "mbti2": self.mbti_dict[mem2],
                    "vibe": vibe_value
                }
        return vibe_result


if __name__ == '__main__':
    data = {"wasabiihater": "ESFJ",
            "junseo.0_0": "ENTP",
            "nimuyx": "ESFJ",
            "_belle.mxmxnt": "ENTJ"}

    # 각 멤버의 MBTI 유형 정보를 딕셔너리 형태로 저장합니다.
    member_mbti_dict = {}
    for key, value in data.items():
        member_mbti_dict[key] = MBTI.get(value)
    # member_mbti_dict = {
    #     'member1': 1,
    #     'member2': 4,
    #     'member3': 12,
    #     'member4': 15,
    #     'member5': 16
    # }

    # MbtiAnalysis 클래스 객체를 생성하여, 각 멤버별 분위기(vibe) 점수를 분석합니다.
    mbti_analysis = MbtiAnalysis(member_mbti_dict)
    mbti_analysis.print_vibe()
    for i in range(4):
        print(i)
        for j in range(i+1, 4):
            print(mbti_analysis.mbti_distance_matrix[i][j])

