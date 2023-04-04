import numpy as np

if __name__ == '__main__':
    a = np.zeros((2, 2))
    b = np.random.random((2, 2))
    b[a == 0] = 0
    print(b)