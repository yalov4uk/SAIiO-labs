from collections import deque
from math import floor

import numpy as np

from dual_simplex import dual_simplex_method, NoPlansException

counter = 0
record_changed = 0


def is_int(value, eps=10 ** -5):
    return abs(int(round(value)) - value) < eps


def solve(A, c, d_low, d_high, b, j_base, int_indices, record=-np.inf):
    lp_list = deque()
    lp_list.append((d_low, d_high, j_base))
    is_int_vectorized = np.vectorize(is_int)

    R = record
    has_best_plan = False
    best_plan = None
    global counter, record_changed
    counter = 0

    while len(lp_list) != 0:
        counter += 1
        lower_bound, upper_bound, base_indices = lp_list.popleft()
        try:
            x, base_indices = dual_simplex_method(A, c, lower_bound, upper_bound, b, base_indices)

            current_value = np.dot(c, x)
            if np.all(is_int_vectorized(x[int_indices])):
                if R < current_value:
                    record_changed += 1
                    R = current_value
                    has_best_plan = True
                    best_plan = x
                    continue

            if current_value < R:
                continue

            non_int_index = np.argwhere(is_int_vectorized(x) == False)[0]

            new_lower_bound = lower_bound.copy()
            new_upper_bound = upper_bound.copy()
            new_upper_bound[non_int_index] = floor(x[non_int_index])
            new_lower_bound[non_int_index] = floor(x[non_int_index]) + 1

            lp_list.append((lower_bound, new_upper_bound, base_indices))
            lp_list.append((new_lower_bound, upper_bound, base_indices))

        except NoPlansException:
            continue

    if has_best_plan is False:
        raise Exception('Solutions not found')
    return R, best_plan


def main():
    A = np.array([
        [1, 0, 1, 0, 4, 3, 4],
        [0, 1, 2, 0, 55, 3.5, 5],
        [0, 0, 3, 1, 6, 2, -2.5],
    ])

    b = np.array([26, 185, 32.5])

    c = np.array([1, 2, 3, -1, 4, -5, 6])

    d_low = np.array([0, 1, 0, 0, 0, 0, 0], dtype=np.int64)
    d_high = np.array([1, 2, 5, 7, 8, 4, 2], dtype=np.int64)

    jb = [0, 1, 2]

    int_req = np.array(range(A.shape[1]), dtype=np.int32)
    R, best_plan = solve(A, c, d_low, d_high, b, jb, int_req)
    print 'x = {}'.format(map(int, map(round, best_plan)))
    print 'r = {}'.format(int(round(R)))
    print 'iteration count = {}'.format(counter)
    print record_changed


if __name__ == '__main__':
    main()
