"""
Implements Lanczos algorithm for producing
a similar tridiagonal matrix to a given matrix
See https://en.wikipedia.org/wiki/Lanczos_algorithm for more info

This has methods that are meant to be imported.
If run directly, it will run tests on a few small matrices
"""

import numpy as np

def lanczos_dense(A, m=None):
    """
    Runs lanczos algorithm
    :param A: the SYMMETRIC matrix to diagonalize
    :param m: the number of alpha/beta parameters to produce.
        If None, it defaults to the side length of A
    """
    
    # Make sure A is a matrix
    assert len(A.shape) == 2
    assert A.shape[0] == A.shape[1]
    n = A.shape[0]
    if m is None:
        m = n

    # Initialize matrices
    alphas = np.zeros(m)
    betas = np.zeros(m-1)
    V = np.zeros((n, m))

    # First iteration
    V[0,0] = 1. # the first vector is [1, 0, 0, ..., 0]
    w1p = np.matmul(A, V[:, 0])
    alphas[0] = np.dot(w1p, V[:, 0])
    w_last = w1p - alphas[0] * V[:, 0]

    for j in range(1, m):
        beta_j = np.linalg.norm(w_last)

        # Choose a new vector vj
        if np.isclose(beta_j, 0):
            vj = np.random.randn(n)
            for i in range(j):
                vj -= np.dot(vj, V[:, i]) * V[:, i]
            vj_norm = np.linalg.norm(vj)
            assert not np.isclose(vj_norm, 0)
            vj /= vj_norm
        else:
            vj = w_last / beta_j
        V[:, j] = vj
        betas[j-1] = beta_j
        wjp = np.matmul(A, V[:, j])
        alphas[j] = np.dot(wjp, V[:, j])
        w_last = wjp - alphas[j] * V[:, j] - betas[j-1] * V[j-1]

    return dict(V=V, alphas=alphas, betas=betas)

def make_dense_T_matrix(alphas, betas):
    m = len(alphas)
    T = np.zeros((m, m))
    T[0, 0] = alphas[0]
    for i in range(1, m):
        T[i, i] = alphas[i]
        T[i-1, i] = betas[i-1]
        T[i, i-1] = betas[i-1]
    return T

if __name__ == "__main__":

    # Run some testing
    print("Running a few tests, should raise assertion error if tests fail")
    A_list = [np.eye(4), np.eye(5) + 3, np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])]
    for test_n, A in enumerate(A_list):
        print("STARTING TEST {}".format(test_n))
        N = len(A)
        print("A:")
        print(A)
        result = lanczos_dense(A)
        V = result["V"]
        alphas = result["alphas"]
        betas = result["betas"]
        print(result)
        T = make_dense_T_matrix(alphas, betas)
        print("T Matrix")
        print(T)

        # Check if V is orthogonal
        mult_by_T = V.T @ V
        print("\nMultiply V by its transpose")
        #print(mult_by_T)
        error = np.linalg.norm(mult_by_T - np.eye(N))
        print("Should be 0: {:.2e}".format(error))
        assert np.isclose(error, 0)

        # Make sure eigenvalues match
        orig_eigvals = np.linalg.eigvalsh(A) 
        T_eigvals = np.linalg.eigvalsh(T)
        print("Orig eigvals: {}".format(orig_eigvals))
        print("T eigvals: {}".format(T_eigvals))
        assert np.allclose(orig_eigvals, T_eigvals)
        
        print("Random vector check")
        i = np.random.randint(len(alphas))
        x1 = np.matmul(A, V[:, i])
        x2 = alphas[i] * V[:, i]
        if i > 0:
            x2 += betas[i-1] * V[:, i-1]
        if i < len(alphas) - 1:
            x2 += betas[i] * V[:, i+1]
        diff_mag = np.linalg.norm(x1 - x2)
        print("This value should be 0:")
        print(diff_mag)
        assert np.isclose(diff_mag, 0)
        print("Test passed")
        print(("#"*20+"\n")*5+"\n"*5)

