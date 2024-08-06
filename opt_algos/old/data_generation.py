#######################
#   Data generation   #
#######################

'''
Our toy example involves a function z=f(x,y). The "true" function is a Branin function
(https://www.sfu.ca/~ssurjano/branin.html), and we add normal noise with a variance
of 120.

The domain of x will be [-5, 10] and the domain of y will be [0, 15].

We generate 8 "synthetic" datasets - datasets A-D will be generated as above, and
datasets E-H will shuffle the z column and multiply it by 20, thus making these
datasets completely useless for training.

Each dataset will contain different ranges of X and Y

                  | X \in [-5, 2.5]  |  X \in [2.5, 10]  |
------------------|------------------|-------------------|
Y \in [0, 7.5]    |     A and E      |      B and F      |
Y \in [7.5, 15]   |     C and G      |      D and H      |
----------------------------------------------------------

Thus, an optimal mix for training would be 0.25 each of A, B, C, and D.

Dataset I will contain the full range of X and Y and not be shuffled - it will act as
a validation set.
'''

import numpy as np
import pandas as pd

def branin(x, y):
    '''
    Return the branin function value at x, y, which we use as our data generating
    pattern
    '''

    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)
    
    return (y - b * x**2 + c * x - r)**2 + s * (1 - t) * np.cos(x) + s

def generate_data():
    '''
    Generate the 9 datasets as described above. This function returns a dict out
    with entries A through I constaining the datasets described above.
    '''

    out = {}

    # Datasets A-H
    # ------------

    x = np.linspace(-5, 10, 200)
    y = np.linspace(0, 15, 200)
    X, Y = [i.ravel() for i in np.meshgrid(x, y)]
    Z = branin(X, Y) + np.random.normal(0, 120, len(X))
    df = pd.DataFrame(np.vstack([X, Y, Z]).T, columns=['X', 'Y', 'Z'])

    out['A'] = df[(df.X <= 2.5) & (df.Y <= 7.5)]
    out['B'] = df[(df.X > 2.5) & (df.Y <= 7.5)]
    out['C'] = df[(df.X <= 2.5) & (df.Y > 7.5)]
    out['D'] = df[(df.X > 2.5) & (df.Y > 7.5)]

    for i, j in zip(['A', 'B', 'C', 'D'], ['E', 'F', 'G', 'H']):
        out[j] = out[i].copy()
        out[j].Z = np.random.choice(out[j].Z, len(out[j]), replace=False)*20
    
    # Dataset I
    # ---------

    x = np.linspace(-5, 10, 100)
    y = np.linspace(0, 15, 100)
    X, Y = [i.ravel() for i in np.meshgrid(x, y)]
    Z = branin(X, Y) + np.random.normal(0, 120, len(X))
    out['I'] = pd.DataFrame(np.vstack([X, Y, Z]).T, columns=['X', 'Y', 'Z'])

    # Return
    # ------
    return out