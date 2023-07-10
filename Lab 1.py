import numpy as np

class TextbookStack(object):
    """ A class that tracks the """
    def __init__(self, initial_order, initial_orientations):
        assert len(initial_order) == len(initial_orientations)
        self.num_books = len(initial_order)
        
        for i, a in enumerate(initial_orientations):
            assert i+1 in initial_order  # 인덱스는 1부터 시작하므로 i+1로 수정
            assert a == 1 or a == 0

        self.order = np.array(initial_order)
        self.orientations = np.array(initial_orientations)

    def flip_stack(self, position):
        assert position <= self.num_books
        
        self.order[:position] = self.order[:position][::-1]
        self.orientations[:position] = np.abs(self.orientations[:position] - 1)[::-1]

    def check_ordered(self):
        for idx, front_matter in enumerate(self.orientations):
            if (idx+1 != self.order[idx]) or (front_matter != 1):  # 인덱스는 1부터 시작하므로 idx+1로 수정
                return False

        return True

    def copy(self):
        return TextbookStack(self.order, self.orientations)
    
    def __eq__(self, other):
        if isinstance(other, TextbookStack):
            return all(self.order == other.order) and all(self.orientations == other.orientations)
        return False

    def __str__(self):
        return f"TextbookStack:\n\torder: {self.order}\n\torientations:{self.orientations}"


def apply_sequence(stack, sequence):
    new_stack = stack.copy()
    for flip in sequence:
        new_stack.flip_stack(flip)
    return new_stack

initial_order = [1, 2, 3, 4]
initial_orientations = [0,1,0,1]
stack = TextbookStack(initial_order, initial_orientations)
print(stack)

sequence = [2, 3]
new_stack = apply_sequence(stack, sequence)
print(new_stack)
