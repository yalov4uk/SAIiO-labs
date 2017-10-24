import numpy as np
from numpy.linalg import inv

EPS = -10 ** -4


class NoPlansException(Exception):
    pass


def dual_simplex_method(A, c, d_low, d_high, b, j_base, j=None, A_base_inv=None):
    A_base = A[:, j_base]
    if A_base_inv == None:
        A_base_inv = inv(A_base)

    if j is None:
        j = set(range(A.shape[1]))

    # step 1
    y = np.dot(c[j_base], A_base_inv)

    delta = np.dot(y, A) - c

    j_non_base = np.array(list(sorted(j - set(j_base))))
    # print np.argwhere(delta[j_non_base] >= -EPS)
    not_zero_indices = np.argwhere(delta[j_non_base] >= -EPS).ravel()
    if len(not_zero_indices) > 0:
        j_non_base_plus = j_non_base[not_zero_indices]
    else:
        j_non_base_plus = np.array([], dtype=int)

    j_non_base_minus = np.array(list(sorted(set(j_non_base) - set(j_non_base_plus))), dtype=int)

    while True:
        # step 2
        kappa = np.zeros((A.shape[1],))
        kappa[j_non_base_plus] = d_low[j_non_base_plus]
        kappa[j_non_base_minus] = d_high[j_non_base_minus]

        kappa[j_base] = np.dot(A_base_inv,
                               b - np.dot(A[:, j_non_base], kappa[j_non_base]))

        # step 3
        if (d_low[j_base] + EPS <= kappa[j_base]).all() and \
                (kappa[j_base] <= d_high[j_base] - EPS).all():
            return kappa, j_base

        # step 4
        not_in_lower_bound_indices = np.ravel(np.argwhere(kappa[j_base] < d_low[j_base] + EPS))
        not_in_higher_bound_indices = np.ravel(np.argwhere(kappa[j_base] > d_high[j_base] - EPS))

        place_in_j_base = list(sorted(set(not_in_higher_bound_indices) |
                                      set(not_in_lower_bound_indices)))[0]
        j_k = j_base[place_in_j_base]
        # step 5
        mu = np.zeros((A.shape[1],))
        mu[j_k] = 1
        if kappa[j_k] > d_high[j_k] - EPS:
            mu[j_k] = -1

        # k is a place of jk of invalid inex in basis
        # k = np.argwhere(j_base == j_base[j_k])
        delta_y = np.dot(mu[j_k], A_base_inv[place_in_j_base, :])
        mu[j_non_base] = np.dot(delta_y, A[:, j_non_base])

        # step 6

        sigma = np.array([
            -delta[j_] / mu[j_]
            if (j_ in j_non_base_plus and mu[j_] < EPS) \
               or (j_ in j_non_base_minus and mu[j_] > -EPS)
            else
            np.inf
            for j_ in j_non_base
        ], dtype=np.float64)

        min_index_in_non_base = np.argmin(sigma)
        sigma0 = sigma[min_index_in_non_base]
        j_min = j_non_base[min_index_in_non_base]

        # 7
        if sigma0 == np.inf:
            raise NoPlansException('No plans')

        # 8

        # delta = np.zeros((A.shape[1],))
        delta = delta + sigma0 * mu

        # 9

        j_base[place_in_j_base] = j_min
        j_base = np.sort(j_base)

        A_base = A[:, j_base]
        A_base_inv = inv(A_base)

        j_non_base = np.array(list(sorted(j - set(j_base))))
        if mu[j_k] == 1 and j_min in j_non_base_plus:
            j_non_base_plus = np.array(
                list(sorted((set(j_non_base_plus) - set([j_min]))
                            | set([j_k])))
                , dtype=int
            )
        elif mu[j_k] == -1 and j_min in j_non_base_plus:
            j_non_base_plus = np.array(
                list(sorted(set(j_non_base_plus) - set([j_min])))
                , dtype=int
            )
        elif mu[j_k] == 1 and j_min not in j_non_base_plus:
            j_non_base_plus = np.array(
                list(sorted(set(j_non_base_plus) | set([j_k])))
                , dtype=int
            )
        elif mu[j_k] == -1 and j_min not in j_non_base_plus:
            j_non_base_plus = j_non_base_plus.copy()

        j_non_base_minus = np.array(list(sorted(set(j_non_base) - set(j_non_base_plus))), dtype=int)


def main():
    # 1
    # c = np.array([3, 2,  0, 3, -2, -4], dtype=float)
    # d_low = np.array([0, -1, 2, 1, -1, 0], dtype=float)
    # d_high = np.array([2, 4, 4, 3, 3, 5], dtype=float)
    #
    #
    # A = np.array([
    #     [2, 1, -1, 0, 0, 1],
    #     [1, 0, 1, 1, 0, 0],
    #     [0, 1, 0, 0, 1, 0]
    # ], dtype=float)
    #
    # b = np.array([2, 5, 0], dtype=float).T
    #
    # j_base = np.array([3, 4, 5])
    # print dual_simplex_method(A,c, d_low, d_high,b , j_base)
    #
    # # 2
    # A = np.array([
    # [1, -5, 3, 1, 0, 0],
    # [4, -1, 1, 0, 1, 0],
    # [2, 4, 2, 0, 0, 1]
    # ], dtype=np.float32)
    #
    # b = np.array([-7, 22, 30], dtype=np.float32)
    # c = np.array([7, -2, 6, 0, 5, 2], dtype=np.float32)
    #
    # d_low = np.array([2, 1, 0, 0, 1, 1])
    # d_high = np.array([6, 6, 5, 2, 4, 6])
    #
    # j_base = np.array([3, 4, 5])
    # print dual_simplex_method(A,c, d_low, d_high,b, j_base)

    # 3
    # A = np.array([
    # [1, 0, 2, 2, -3, 3],
    # [0, 1, 0, -1, 0, 1],
    # [1, 0, 1, 3, 2, 1]
    # ], dtype=np.float32)
    #
    # b = np.array([15, 0, 13], dtype=np.float32)
    # c = np.array([3, 0.5, 4, 4, 1, 5], dtype=np.float32)
    #
    # d_low = np.array([0, 0, 0, 0, 0, 0])
    # d_high = np.array([3, 5, 4, 3, 3, 4])
    #
    # j_base = np.array([3, 4, 5])
    # print dual_simplex_method(A, c, d_low, d_high, b, j_base)

    # 4

    # A = np.array([
    #     [1, 0, 0, 12, 1, -3, 4, -1],
    #     [0, 1, 0, 11, 12, 3, 5, 3],
    #     [0, 0, 1, 1, 0, 22, -2, 1]
    # ], dtype=float)
    #
    # b = np.array([40, 107, 61], dtype=float)
    # c = np.array([2, 1, -2, -1, 4, -5, 5, 5], dtype=float)
    #
    # d_low = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    # d_high = np.array([3, 5, 5, 3, 4, 5, 6, 3], dtype=float)
    #
    # j_base = np.array([0, 1, 3])
    # x, base = dual_simplex_method(A, c, d_low, d_high, b, j_base)
    # print x
    # print np.dot(c,x)

    # 5

    # A = np.array([
    #     [1, -3, 2, 0, 1, -1, 4, -1, 0],
    #     [1, -1, 6, 1, 0, -2, 2, 2, 0],
    #     [2, 2, -1, 1, 0, -3, 8, -1, 1],
    #     [4, 1, 0, 0, 1, -1, 0, -1, 1],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1]
    # ], dtype=np.float32)
    #
    # b = np.array([3, 9, 9, 5, 9], dtype=np.float32)
    # c = np.array([-1, 5, -2, 4, 3, 1, 2, 8, 3], dtype=np.float32)
    #
    # d_low = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0],dtype=float)
    # d_high = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5], dtype=float)
    #
    # j_base = np.array([3, 4, 5, 6, 7])


    # no plans
    # A = np.array([
    #     [1, 7, 2, 0, 1, -1, 4],
    #     [0, 5, 6, 1, 0, -3, -2],
    #     [3, 2, 2, 1, 1, 1, 5]
    # ], dtype=np.float32)
    #
    # b = np.array([1, 4, 7], dtype=np.float32)
    # c = np.array([1, 2, 1, -3, 3, 1, 0], dtype=np.float32)
    #
    # d_low = np.array([-1, 1, -2, 0, 1, 2, 4])
    # d_high = np.array([3, 2, 2, 5, 3, 4, 5])
    #
    # j_base = np.array([3, 4, 5])

    # 1.5
    # A = np.array([
    #     [2, -1, 1, 0, 0, -1, 3],
    #     [0, 4, -1, 2, 3, -2, 2],
    #     [3, 1, 0, 1, 0, 1, 4]
    # ], dtype=np.float32)
    #
    # b = np.array([1.5, 9, 2], dtype=np.float32)
    # c = np.array([0, 1, 2, 1, -3, 4, 7], dtype=np.float32)
    #
    # d_low = np.array([0, 0, -3, 0, -1, 1, 0])
    # d_high = np.array([3, 3, 4, 7, 5, 3, 2])
    #
    # j_base = np.array([3, 4, 5])


    # 37.555
    A = np.array([
        [1, 3, 1, -1, 0, -3, 2, 1],
        [2, 1, 3, -1, 1, 4, 1, 1],
        [-1, 0, 2, -2, 2, 1, 1, 1]
    ], dtype=np.float32)

    b = np.array([4, 12, 4], dtype=np.float32)
    c = np.array([2, -1, 2, 3, -2, 3, 4, 1], dtype=np.float32)

    d_low = np.array([-1, -1, -1, -1, -1, -1, -1, -1])
    d_high = np.array([2, 3, 1, 4, 3, 2, 4, 4])

    j_base = np.array([3, 4, 5])

    A = np.array([
        [-2, -1, 1, 0, 0],
        [3, 1, 0, 1, 0],
        [-1, 1, 0, 0, 1]
    ])

    b = np.array([-1, 10, 3])
    c = np.array([2, 0, 0, 0, -5])
    d_low = np.array([0, 0, 0, 0, 0])

    inf = 1e32
    d_high = np.array([inf, inf, inf, inf, inf])

    j_base = np.array([0, 1, 2], dtype=int)

    x, base = dual_simplex_method(A, c, d_low, d_high, b, j_base)
    print x
    print np.dot(c, x)


if __name__ == '__main__':
    main()
