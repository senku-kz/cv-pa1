import numpy as np


def dot_product(vector1, vector2):
    """ Implement dot product of the two vectors.
    Args:
        vector1: numpy array of shape (n, 1)
        vector2: numpy array of shape (n, 1)

    Returns:
        out: scalar value
    >>> dot_product(np.array([[1],[2]]), np.array([[4],[8]]))
    array([[20]])
    >>> dot_product(np.array([[1],[2],[-5]]), np.array([[4],[8],[1]]))
    array([[15]])
    >>> dot_product(np.array([[1],[2],[-5],[2]]), np.array([[4],[8],[1],[-2]]))
    array([[11]])
    """
    out = None
    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    if vector1.shape[0] == vector2.shape[0] and vector1.shape[1] == vector2.shape[1]:
        out = np.dot(vector1.T, vector2)
    elif vector1.shape[0] == vector2.shape[1] and vector1.shape[1] == vector2.shape[0]:
        out = np.dot(vector1, vector2)
    else:
        out = None
    ######################################
    #        END OF YOUR CODE            #
    ######################################
    return out


def matrix_mult(M, vector1, vector2, vector3):
    """ Implement (vector2.T * vector3) * (M * vector1.T)
    Args:
        M: numpy matrix of shape (m, n)
        vector1: numpy array of shape (1, n)
        vector2: numpy array of shape (n, 1)
        vector3: numpy array of shape (n, 1)

    Returns:
        out: numpy matrix of shape (m, 1)

    >> matrix_mult( np.array([[0, 1, 2],[3, 4, 5],[6, 7, 8],[9, 10, 11]])\
                    ,np.array([[1, 1, 0]])\
                    ,np.array([[-1],[2],[5]])\
                    ,np.array([[2],[2],[0]])\
        )
    array([[2], [14], [26], [38]])
    """
    out = None
    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    out = np.dot(vector2.T, vector3) * np.dot(M, vector1.T)
    ######################################
    #        END OF YOUR CODE            #
    ######################################
    return out


def svd(matrix):
    """ Implement Singular Value Decomposition
    Args:
        matrix: numpy matrix of shape (m, n)

    Returns:
        u: numpy array of shape (m, m)
        s: numpy array of shape (k)
        v: numpy array of shape (n, n)
    """
    u = None
    s = None
    v = None
    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    u, s, v = np.linalg.svd(matrix, full_matrices=True)
    ######################################
    #        END OF YOUR CODE            #
    ######################################
    return u, s, v


def get_singular_values(matrix, n):
    """ Return top n singular values of matrix
    Args:
        matrix: numpy matrix of shape (m, w)
        n: number of singular values to output
        
    Returns:
        singular_values: array of shape (n)
    >> get_singular_values(np.array([[0, 1, 2],[3, 4, 5],[6, 7, 8],[9, 10, 11]]), 1)
    22.44674882
    >> get_singular_values(np.array([[0, 1, 2],[3, 4, 5],[6, 7, 8],[9, 10, 11]]), 2)
    1.4640585017492267
    """
    singular_values = None
    u, s, v = svd(matrix)
    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    singular_values = s[:n]
    ######################################
    #        END OF YOUR CODE            #
    ######################################
    return singular_values
