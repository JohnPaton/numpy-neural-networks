import numpy as np


# #### Data functions
def rescale(X):
    """Scale dataset X to zero mean and unit variance"""
    return (X - X.mean(axis=0)) / (X.var(axis=0))


def division_sign_dataset(n_samples, seed=None):
    """Generate the division sign shaped dataset and labels"""
    if seed:
        np.random.seed(seed)
    X0 = np.random.normal(loc=[0, 0], scale=[2, 0.5],
                          size=(int(n_samples / 2), 2))
    X11 = np.random.normal(loc=[0, 4], scale=[0.5, 1],
                           size=(int(n_samples / 4), 2))
    X12 = np.random.normal(loc=[0, -4], scale=[0.5, 1],
                           size=(int(n_samples / 4), 2))
    X1 = np.vstack([X11, X12])
    X = np.vstack([X0, X1])

    X = rescale(X)

    y0 = np.zeros(shape=(int(n_samples / 2), 1))
    y1 = np.ones(shape=(int(n_samples / 2), 1))
    yhat = np.vstack([y0, y1])

    # shuffle the data
    inds = np.random.permutation(np.arange(n_samples))
    X = X[inds]
    yhat = yhat[inds]

    return X, yhat


# #### Activation functions
def relu(z):
    return np.where(z > 0, z, 0)

    
def drelu_dz(z):
    return np.where(z > 0, 1, 0)


def sig(z):
    return 1 / (1 + np.exp(-z))


def dsig_dz(z):
    return sig(z) * (1 - sig(z))


# #### Loss functions
def J(y, yhat):
    eps = 1e-8
    return -(yhat * np.log(y + eps) + (1 - yhat) * np.log(1 - y + eps))


def dJ_dy(y, yhat):
    eps = 1e-8
    return (1 - yhat) / (1 - y + eps) - yhat / (y + eps)


def Jreg(y, yhat, w, lam):
    eps = 1e-8
    cost_term = -(yhat * np.log(y + eps) + (1 - yhat) * np.log(1 - y + eps))
    reg_term = 0.5 * lam * sum([(w[l]**2).sum() for l in w])
    return cost_term + reg_term


def dJreg_dy(y, yhat, w, lam):
    eps = 1e-8
    return (1 - yhat) / (1 - y + eps) - yhat / (y + eps)


# #### Model functions
def init_model(shape, seed=None):
    if seed:
        np.random.seed(seed)

    w = {}
    b = {}
    for layer in range(1, len(shape)):  # no weights for the input layer
        # dim: from_units, to_units
        w[layer] = np.random.normal(0, 0.5, size=(shape[layer - 1], shape[layer]))
        # dim: to_units, 1
        b[layer] = np.random.normal(0, 0.5, size=(shape[layer], 1))

    return w, b
