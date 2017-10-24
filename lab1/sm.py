from numpy import *


class SimplexMethodSolver(object):
    def __init__(self, A, b, c, x=None, basic_indexes=None, nonbasic_indexes=None):
        self.A = A
        self.b = b
        self.c = c
        self.x = x
        self.m, self.n = A.shape
        self.basic_indexes = basic_indexes
        self.nonbasic_indexes = nonbasic_indexes
        self.eps = finfo(float).eps

    def is_zero(self, value):
        abs(value) < self.eps

    def normalize(self):
        if (self.b < 0).any():
            indexes = self.b < 0
            self.b[indexes] *= -1
            self.A[indexes, :] *= -1

    def first_phase(self):
        self.normalize()
        new_A = append(self.A.transpose(), eye(self.m, dtype=float64), axis=0).transpose()
        x = append(zeros(self.n, dtype=float64), self.b)
        c = append(zeros(self.n, dtype=float64), -ones(self.m, dtype=float64))

        nonbasic_indexes = range(self.n)
        basic_indexes = range(self.n, self.n + self.m)

        J = set(nonbasic_indexes)
        Ju = set(basic_indexes)

        result_x = self.second_phase(new_A, self.b, c, x, basic_indexes, nonbasic_indexes)

        if not all(abs(result_x[-self.m:]) < self.eps):
            raise Exception("This task has no solution, because it restrictions is not compatible")

        B = linalg.inv(new_A[:, basic_indexes])
        lost_indexes = list(J - set(basic_indexes))

        while True:
            if not set(basic_indexes) & set(Ju):
                lJ = list(J)
                self.A = new_A[:, lJ]
                self.c = self.c[lJ]
                self.x = result_x[lJ]
                self.basic_indexes = [lJ.index(el) for el in basic_indexes]
                self.nonbasic_indexes = [i for i in lJ if i not in self.basic_indexes]
                return self

            jk = (set(basic_indexes) & set(Ju)).pop()
            k = jk - self.n
            ek = eye(self.m)[:, k]

            tmp = dot(ek, B)
            alpha = dot(tmp, new_A[:, lost_indexes])

            if not all(abs(alpha) < self.eps):
                s = list(abs(alpha) > self.eps).index(True)
                js = lost_indexes[s]
                basic_indexes[k] = js
            else:
                del basic_indexes[k]
                Ju.remove(jk)
                new_A = delete(new_A, k, axis=0)
                self.b = delete(self.b, k)
                B = delete(B, k, axis=0)
                B = delete(B, k, axis=1)
                self.m -= 1

    def get_index(self, a, el):
        try:
            return a.index(el)
        except ValueError:
            return -1

    def change_B(self, z, s, m):
        zk = z[s]
        z[s] = -1
        z /= -zk
        M = eye(m)
        M[:, s] = z
        self.B = dot(M, self.B)

    def second_phase(self, A, b, c, x, basic_indexes, nonbasic_indexes):
        m, n = A.shape
        basic_a = A[:, basic_indexes]
        self.B = linalg.inv(basic_a)
        while True:
            basic_c = array([c[i] for i in basic_indexes])

            # Create potential and estimate vectors
            u = dot(basic_c, self.B)
            delta = array(subtract(dot(u, A), c))

            k = self.get_index([delta[j] < 0 and not self.is_zero(delta[j]) for j in nonbasic_indexes], True)
            if not ~k:
                return x
            j0 = nonbasic_indexes[k]

            z = dot(self.B, A[:, j0])

            if all(z <= self.eps):
                raise ValueError("This task has no solution, because her target function is not limited at plans set")

            basic_x = x[basic_indexes]

            tetta = [basic_x[j] / z[j] if z[j] > 0 and not self.is_zero(z[j]) else inf for j in xrange(m)]
            tetta0 = min(tetta)
            s = self.get_index(tetta, tetta0)
            index_tetta0 = basic_indexes[s]

            for i, j in enumerate(basic_indexes):
                x[j] = x[j] - tetta0 * z[i]
            x[j0] = tetta0

            basic_indexes[s] = j0
            basic_a[:, s] = A[:, j0]
            self.change_B(z, s, m)

            nonbasic_indexes[k] = index_tetta0

    def solve(self):
        if self.x is None:
            self.first_phase()
        self.x = self.second_phase(self.A, self.b, self.c, self.x, self.basic_indexes, self.nonbasic_indexes)
        return self


if __name__ == '__main__':
    # Variant #2
    a = array([
        [0.0, 1.0, 1.0, 1.0, 0.0, -8.0, 1.0, 5.0],
        [0.0, -1.0, 0.0, -7.5, 0.0, 0.0, 0.0, 2.0],
        [0.0, 2.0, 1.0, 0.0, -1.0, 3.0, -1.4, 0.0],
        [1.0, 1.0, 1.0, 1.0, 0.0, 3.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 0.0, 3.0, 1.0, 1.0]
    ])
    b = array([15.0, -45.0, 1.8, 19.0, 19.0])
    c = array([-6.0, -9.0, -5.0, 2.0, -6.0, 0.0, 1.0, 3.0])

    x = array([4.0, 0.0, 6.0, 6.0, 0.0, 0.0, 3.0, 0.0])

    basic_indexes = [0, 2, 3, 6]
    nonbasic_indexes = [1, 4, 5, 7]

    res = SimplexMethodSolver(a, b, c).solve()
    print res.x
    print res.basic_indexes
