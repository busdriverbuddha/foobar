class Matrix:

    def __init__(self, values):
        self.values = values
        self.rows = len(self.values)
        self.cols = len(self.values[0])

    def __getitem__(self, items):
        if isinstance(items, slice):
            return self[items, :]
        u, v = items
        if isinstance(u, slice) and isinstance(v, slice):
            return Matrix([row[v] for row in self.values[u]])
        elif isinstance(u, slice):
            return Matrix([[row[v]] for row in self.values[u]])
        elif isinstance(v, slice):
            return Matrix([self.values[u][v]])
        else:
            return self.values[u][v]

    def __add__(self, other):
        return Matrix([[self[i,j] + other[i,j] for j in range(self.cols)] for i in range(self.rows)])

    def __sub__(self, other):
        return Matrix([[self[i,j] - other[i,j] for j in range(self.cols)] for i in range(self.rows)])

    def __neg__(self):
        return Matrix([[-v for v in row] for row in self.values])

    def __mul__(self, other):
        if isinstance(other, Matrix):
            return Matrix(
                values=[
                    [
                        sum(self[i,k]*other[k,j] for k in range(self.cols))
                        for j in range(other.cols)
                    ]
                    for i in range(self.rows)
                ]
            )
        else:
            return Matrix([[other*v for v in row] for row in self.values])

    def __str__(self):
        return "\n".join("\t".join(str(v) for v in row) for row in self.values)

    def __repr__(self):
        return "Matrix({:s})".format(str([[str(v) for v in row] for row in self.values]))

    def remove_row(self, n):
        return Matrix([self.values[i] for i in range(self.rows) if i != n])
    
    def remove_col(self, n):
        return Matrix([[row[j] for j in range(self.cols) if j != n] for row in self.values])

    def replace_row(self, i, row):
        self.values[i] = row

    def swap_rows(self, i, j):
        self.values[i], self.values[j] = self.values[j], self.values[i]

    def swap_cols(self, m, n):
        for row in self.values:
            row[m], row[n] = row[n], row[m]

def determinant(M):
    if M.rows == 1:
        return M[0,0]
    return sum(((-1)**i)*M[i,0]*determinant(M.remove_col(0).remove_row(i)) for i in range(M.rows))

def eye(n):
    return Matrix([[1 if i == j else 0 for i in range(n)] for j in range(n)])

def concatenate(A, B):
    return Matrix([A.values[i] + B.values[i] for i in range(A.rows)])

def inverse(A):
    n = A.rows
    M = concatenate(A, eye(n))
    for i in range(n):
        found_pivot = M[i,i] != 0
        if not found_pivot:
            for k in range(i+1, n):
                if M[k,i] != 0:
                    M.swap_rows(i,k)
                    found_pivot = True
                    break
        if not found_pivot:
            break

        M.replace_row(i, (M[i,:] * (1/M[i,i])).values[0])
        for j in range(i+1, n):
            
            M.replace_row(j, (M[j,:] - M[i,:] * M[j,i]).values[0])

    for i in range(1, n):
        for j in range(i):
            M.replace_row(j, (M[j,:] - M[i,:] * M[j,i]).values[0])

    return M[:,n:]

def factor(n):
    F = []
    k = 2
    while n > 1:
        if n % k == 0:
            n //= k
            F.append(k)
        else:
            k += 1
    return {f:F.count(f) for f in F}
        

def lcm(a, b):
    from fractions import gcd
    return a * b // gcd(a, b)

def lcmm(*args):
    from functools import reduce
    return reduce(lcm, args)

    

def solution(m):
    from fractions import Fraction

    if m == [[]]:
        print("empty matrix")
        return

    if sum(m[0][1:]) == 0:
        sol = [1]
        for i, row in enumerate(m[1:], 1):
            if sum(row[:i] + row[i+1:]) == 0:
                sol = sol + [0]

        sol = sol + [1]
        return sol

    if len(m) == 1:
        return [1, 1]

    nonzeros = [i for i, row in enumerate(m) if sum(row[:i] + row[i+1:]) != 0]
    first_end_state = len(nonzeros)
    zeros = [i for i, row in enumerate(m) if sum(row[:i] + row[i+1:]) == 0]
    indices = nonzeros + zeros
    
    P = [[0] * len(m) for _ in range(len(m))]
    for i, k in enumerate(indices):
        for j, l in enumerate(indices):
            P[i][j] = m[k][l]
        
    for i, row in enumerate(P):
        if sum(row) > 0:
            den = sum(row)
            P[i] = [Fraction(v, den) for v in row]


    for i in range(first_end_state, len(m)):
        P[i][i] = 1

    P = Matrix(P)
    

    k = first_end_state

    Q = P[:k,:k]
    R = P[:k,k:]

    N = inverse(eye(Q.rows) - Q)
    
    B = N*R
    probs = B[0,:]
    probs = probs.values[0]

    D = [(v.denominator if isinstance(v, Fraction) else 1) for v in probs]

    from functools import reduce

    final_denominator = lcmm(*D)

    L = [final_denominator // v.denominator * v.numerator for v in probs] + [final_denominator]
    return L
