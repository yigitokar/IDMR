# IDMR
This repo contains the code base for Iterative Distributed Multinomial Regression with examples of how to run it.
### Abstract
This article introduces an iterative estimator for the multinomial logistic regression model that is both asymptotically efficient and fast to compute
even when the number of choices is large. In many economic applications such as text analysis and spatial choice models, the number of discrete choices 
can be large. Solving for the maximum likelihood estimator via traditional optimiza- tion algorithms, such as Newton-Raphson, is infeasible because the
number of arguments in the log likelihood function is enormous. We tackle this problem by proposing an iterative estimator that optimizes the two parts
of the log likelihood function in turn. The proposed estimator allows for distributed com- puting, which substantially reduces the computational time. We
show that the estimator is consistent and has the same asymptotic distribution as the maxi- mum likelihood estimator. Via an extensive simulation study,
we show that the iterative estimator has good finite sample performance and is extremely fast to compute.
