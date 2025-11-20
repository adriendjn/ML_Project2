#!/usr/bin/env python3
from scipy.sparse import *
import numpy as np
import pickle
import random
from os import path


def main():
    print("loading cooccurrence matrix")
    with open(path.join(path.split(__file__)[0],path.normcase("cooc.pkl")), "rb") as f:
        cooc = pickle.load(f)
    print("{} nonzero entries".format(cooc.nnz))

    nmax = 100
    print("using nmax =", nmax, ", cooc.max() =", cooc.max())

    print("initializing embeddings")
    embedding_dim = 20
    xs = np.random.normal(size=(cooc.shape[0], embedding_dim))
    ys = np.random.normal(size=(cooc.shape[1], embedding_dim))

    eta = 0.001
    alpha = 3 / 4

    epochs = 10

    for epoch in range(epochs):
        print("epoch {}".format(epoch))
        for ix, jy, n in zip(cooc.row, cooc.col, cooc.data):
        # fill in your SGD code here,
        # for the update resulting from co-occurence (i,j)
            x_dn = np.log(n)
            f_dn = min(1, (n / nmax)**alpha)
            # Retrieving x and y
            x, y  = xs[ix, :], ys[jy, :]
            # Computing the gradient of the loss
            grad_loss = f_dn * (x_dn - np.dot(x,y))
            scale = eta * grad_loss
            # Updating the vectors
            xs[ix, :] += scale * y
            ys[jy, :] += scale * x
    np.save(path.join(path.split(__file__)[0], path.normcase("embeddings.npy")), xs)


if __name__ == "__main__":
    main()
