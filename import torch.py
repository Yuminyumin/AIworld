# you can add imports but you should not rely on libraries that are not already provided in "requirements.txt #
from collections import deque

def breadth_first_search(stack):
    flip_sequence = []

    # --- v ADD YOUR CODE HERE v --- #
    return flip_sequence
    # ---------------------------- #


def depth_first_search(stack):
    flip_sequence = []

    # --- v ADD YOUR CODE HERE v --- #
    return flip_sequence
    # ---------------------------- #




initial_order = [1, 2, 3, 4]
initial_orientations = [0,1,0,1]
stack = TextbookStack(initial_order, initial_orientations)
print(stack)

sequence = [2, 3]
new_stack = apply_sequence(stack, sequence)
print(new_stack)
