import numpy as np

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
                mbti1 = self.mbti_dict[mem1]
                mbti2 = self.mbti_dict[mem2]
                self.mbti_distance_matrix[i, j] = MBTI_GOODNESS[mbti1-1, mbti2-1]

    def print_vibe(self):
        for i, mem1 in enumerate(self.member):
            for j, mem2 in enumerate(self.member):
                print(f"{mem1}와 {mem2}의 궁합 점수: {self.mbti_distance_matrix[i, j]}")

# 예시 멤버 MBTI 데이터를 사용해 인스턴스를 생성하고 출력합니다.
member_mbti_dict = {
    'member1': 1,
    'member2': 4,
    'member3': 12,
    'member4': 15,
    'member5': 16
}

mbti_analysis = MbtiAnalysis(member_mbti_dict)
mbti_analysis.print_vibe()
