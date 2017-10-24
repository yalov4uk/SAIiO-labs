from numpy import *

from sm import SimplexMethodSolver


def main():
    A = array([
        array([2, -1, 1, 0, 0, -1, 3], dtype=float),
        array([0, 4, -1, 2, 3, -2, 2], dtype=float),
        array([3, 1, 0, 1, 0, 1, 4], dtype=float),
    ])
    b = array([1.5, 9, 2], dtype=float)
    c = array([0, 1, 2, 1, -3, 4, 7], dtype=float)
    d_low = array([0, 0, -3, 0, -1, 1, 0], dtype=float)
    d_high = array([3, 3, 4, 7, 5, 3, 2], dtype=float)
    j_basic = None  # [3, 4, 5]
    j_non_basic = None  # [0, 1, 2]
    x0 = DualSimplexMethodSolver(A, b, c, d_low, d_high, j_basic, j_non_basic).solve()
    if x0.is_solved:
        print 'x0 = ', x0.x
        print "c'x0 = ", dot(c, x0.x)


class DualSimplexMethodSolver(object):
    def __init__(self, A, b, c, d_low, d_high, basic_indexes=None, nonbasic_indexes=None):
        self.is_solved = False
        self.A = A
        self.b = b
        self.c = c
        self.d_low = d_low
        self.d_high = d_high
        self.m, self.n = A.shape
        self.basic_indexes = basic_indexes
        self.nonbasic_indexes = nonbasic_indexes
        self.eps = 0.0000001

    def is_zero(self, val):
        return abs(val) < self.eps

    def init_indexes(self):
        if self.basic_indexes is None:
            plan = SimplexMethodSolver(self.A, self.b, self.c).first_phase()
            self.A = plan.A
            self.b = plan.b
            self.c = plan.c
            self.m = plan.m
            self.n = plan.n
            self.basic_indexes = plan.basic_indexes
            self.nonbasic_indexes = plan.nonbasic_indexes

    def init_auxiliary_variables(self):
        self.basic_A = self.A[:, self.basic_indexes]
        self.B = linalg.inv(self.basic_A)
        self.y = dot(self.c[self.basic_indexes], self.B)
        self.delta = subtract(dot(self.y, self.A), self.c)  # Coplan

    def init_non_basic_indexes(self):
        self.nonbasic_indexes_plus = set(
            [i for i in self.nonbasic_indexes if self.delta[i] > 0 or self.is_zero(self.delta[i])])
        self.nonbasic_indexes_minus = set(self.nonbasic_indexes) - self.nonbasic_indexes_plus

    def init_necessary_params(self):
        self.init_indexes()
        self.init_auxiliary_variables()
        self.init_non_basic_indexes()

    def create_ksi_for_nonbasic_indexes(self):
        for i in range(self.n):
            if i in self.nonbasic_indexes_minus:
                self.ksi.append(self.d_high[i])
            elif i in self.nonbasic_indexes_plus:
                self.ksi.append(self.d_low[i])
            else:
                self.ksi.append(0)

    def create_ksi_for_basic_indexes(self):
        s = reduce(lambda x, y: add(x, y), [dot(self.A[:, j], self.ksi[j]) for j in self.nonbasic_indexes])
        ksi_basic = dot(self.B, subtract(self.b, s))

        for i, index in enumerate(self.basic_indexes):
            self.ksi[index] = ksi_basic[i]

    def create_ksi(self):
        self.ksi = []
        self.create_ksi_for_nonbasic_indexes()
        self.create_ksi_for_basic_indexes()

    def get_index(self, a, el):
        try:
            return list(a).index(el)
        except ValueError:
            return -1

    def calculate_mu(self):
        self.mjk = 1.0 if self.ksi[self.jk] < self.d_low[self.jk] else -1.0
        dy = dot(dot(self.mjk, eye(self.m)[:, self.k]), self.B)
        self.mu = [dot(dy, self.A[:, j]) for j in self.nonbasic_indexes]

    def calculate_sigma0(self):
        def condition(i, j):
            return (j in self.nonbasic_indexes_plus and (self.mu[i] < 0 and not self.is_zero(self.mu[i])) or (
            j in self.nonbasic_indexes_minus and self.mu[i] > 0 and not self.is_zero(self.mu[i])))

        sigma = [-self.delta[j] / self.mu[i] if condition(i, j) else inf for i, j in enumerate(self.nonbasic_indexes)]
        self.sigma0 = min(sigma)
        self.asteriks = sigma.index(self.sigma0)
        self.j_ast = self.nonbasic_indexes[self.asteriks]

    def change_delta(self):
        for i, j in enumerate(self.nonbasic_indexes):
            self.delta[j] += self.sigma0 * self.mu[i]
        self.delta[self.jk] += self.sigma0 * self.mjk

    def change_B(self):
        z = dot(self.B, self.A[:, self.j_ast])
        zk = z[self.k]
        z[self.k] = -1
        z /= -zk
        M = eye(self.m)
        M[:, self.k] = z
        self.B = dot(M, self.B)

    def correct_basic_indexes(self):
        self.basic_indexes[self.k] = self.j_ast
        self.basic_A[:, self.k] = self.A[:, self.j_ast]
        self.change_B()

    def correct_nonbasic_indexes(self):
        self.nonbasic_indexes[self.asteriks] = self.jk
        if self.mjk == 1:
            if self.j_ast in self.nonbasic_indexes_plus:
                self.nonbasic_indexes_plus.remove(self.j_ast)
                self.nonbasic_indexes_plus.add(self.jk)
            else:
                self.nonbasic_indexes_plus.add(self.jk)
        else:
            if self.j_ast in self.nonbasic_indexes_plus:
                self.nonbasic_indexes_plus.remove(self.j_ast)
        self.nonbasic_indexes_minus = set(self.nonbasic_indexes) - self.nonbasic_indexes_plus

    def solve(self):
        self.init_necessary_params()
        while (True):
            self.create_ksi()
            # step 3
            self.jk = self.get_index(logical_and(self.ksi >= self.d_low, self.ksi <= self.d_high), False)
            if not ~self.jk:
                self.is_solved = True
                self.x = array(self.ksi)
                return self
            self.k = self.basic_indexes.index(self.jk)
            # step 4
            self.calculate_mu()
            # step 5
            self.calculate_sigma0()
            if isinf(self.sigma0):
                self.is_solved = False
                print ValueError("This task hasn't possible plans!")
                return self
            self.change_delta()
            self.correct_basic_indexes()
            self.correct_nonbasic_indexes()
            # print self.basic_indexes


if __name__ == '__main__':
    main()
