# Factorization Machine for Prediction
Factorization Machine for regression and classification


**Note 1** : The PyTorch implementation is mine while the Keras and Theano versions were found on the web (references at the top of relevant notebooks).

**Note 2** : The PyTorch implementation uses the efficient O(k.N) formula from Steffen Rendle.

**Note 3** : I am quite new to PyTorch so do not hesitate to highlight all improvements you deem necessary.


#### Goal
Factorization Machine offer an efficient way to include **interactions between pair of variables** in a linear or logistic equation.

When the problem has only a few explanatory variables one can simply extend the linear equation with product variables :

Y = [w1 x1 + w2 x2 + w3 x3 + w4 x4] + [w12 x1.x2 + w13 x1.x3 + w14 x1.x4 + w23 x2.x3 + w24 x2.x4 + w34 x3.x4]

However when the model has 1000s of variables (when is likely to happen with one-hot encoding of categorical variables) this strategy is not optimal. It can still be done (using regularization penalties) tough but Factorization Machines are a good alternative.

Factorization Machine constrain the previous equation to have cross-weights that are the dot-products of feature embedding vectors. Concretely the model searches vectors V1, V2, V3, V4 such that w(i,j) can be replaced by V(i).V(j) :

Y = [w1 x1 + w2 x2 + w3 x3 + w4 x4] + SUM(i,j)OF[ dotprod(V(i),V(j)) x(i).x(j) ]

Instead of (0.5 * N^2) cross-weights the model "only" has to find N embedding vectors.


#### Some great links about Factorization Machine for binary classification
- http://tech.adroll.com/blog/data-science/2015/08/25/factorization-machines.html
- http://www.algo.uni-konstanz.de/members/rendle/pdf/Rendle2010FM.pdf
- https://arxiv.org/abs/1701.04099 (Field-aware Factorization Machines in a Real-world Online Advertising System)
- http://research.criteo.com/ctr-prediction-linear-model-field-aware-factorization-machines/


#### Code and libraries
- http://libfm.org/
