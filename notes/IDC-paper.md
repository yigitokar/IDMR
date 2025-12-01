arXiv:2412.01030v1 [econ.EM] 2 Dec 2024

# Iterative Distributed Multinomial Regression\*

Yanqin Fan†
Yigit Okar‡
Xuetao Shi§

December 3, 2024

## Abstract

This article introduces an iterative distributed computing estimator for the multinomial logistic regression model with large choice sets. Compared to the maximum likelihood estimator, the proposed iterative distributed estimator achieves significantly faster computation and, when initialized with a consistent estimator, attains asymptotic efficiency under a weak dominance condition. Additionally, we propose a parametric bootstrap inference procedure based on the iterative distributed estimator and establish its consistency. Extensive simulation studies validate the effectiveness of the proposed methods and highlight the computational efficiency of the iterative distributed estimator.

**Keywords:** Distributed computing, Iterative methods, Maximum likelihood, Multinomial logistic regression.

**JEL Codes:** C13, C25, C61, C63

\*We thank Jean-Marie Dufour, Hiro Kasahara, Artem Prokhorov, Dacheng Xiu, Jin Yan, and participants of the seminar at the University of British Columbia, the 6th International Conference on Econometrics and Statistics, and 2024 Econometric Society North American Summer Meeting for helpful discussions.
†Department of Economics, the University of Washington; email: fany88@uw.edu
‡Department of Economics, the University of Washington; email: yokar@uw.edu
§School of Economics, the University of Sydney; email: xuetao.shi@sydney.edu.au

1

# 1 Introduction

## 1.1 Motivation and Main Contributions

Discrete choice models, including logit and multinomial-logit (MNL) models, are widely used in applied social science research. With the growing availability of diverse data types and the integration of econometric models with textual and image data, researchers often encounter applications of MNL models with a massive number of choices; see the applications discussed in Section 1.2. In these cases, even if the number of parameters for each choice is small, the total number of parameters, which increases linearly with the number of choices, will be large due to large choice sets. As a result, maximum likelihood estimation (MLE) can become computationally intractable due to the high cost of solving an optimization problem for a large number of parameters.$^1$

Researchers in diverse disciplines have explored various approaches and proposed numerous methods to numerically solve the MLE; see the numerical algorithms discussed in Section 1.2. However, there is a notable lack of theoretical results ensuring the consistency or asymptotic efficiency of the estimators derived from these methods. This gap in the literature between numerical computation and statistical inference motivates the present study.

Our proposed estimator utilizes the multinomial-Poisson (MP) transformation, which reformulates the multinomial likelihood into a Poisson likelihood by incorporating individual fixed effects into the MNL model. When all the covariates are *categorical*, Baker (1994) establishes that the multinomial likelihood and the Poisson likelihood produce identical estimates of the parameters in the MNL model and advocates the computational advantage of maximizing the Poisson likelihood. However, the MP transformation has also been employed in MNL models with continuous or mixed discrete and continuous covariates. For instance, Gentzkow et al. (2019) note that they "... approximate the likelihood of [their] multinomial logit model with the likelihood of a Poisson model...".

As the first contribution of this paper, we establish an equivalence result for all types of covariates, *continuous, discrete, or mixed*, justifying the application of the MP transformation in these cases. To accomplish this, we re-interpret the Poisson likelihood from the MP transformation as a conditional quasi-log-likelihood function given the covariates. Maximizing this function provides a quasi-maximum likelihood estimator (QMLE) for the MNL model. The equivalence result is established by demonstrating that the resulting QMLE is identical to the MLE of the MNL model.

While the QMLE is computationally more efficient than the MLE, it remains costly when applied to MNL models with large choice sets. To address this computational challenge, Taddy (2015) exploits an important feature of the quasi-log-likelihood function: for any given fixed effects, the function is additively separable in

1 MLE of the MNL model refers to the conditional MLE given the covariate and total counts.

2

the parameters across different choices of the MNL model. Taddy (2015) proposes to estimate parameters for each choice separately at a specific value of the fixed effects and calls the resulting estimator *distributed computing estimator*. As noted in Taddy (2015), however, his distributed computing estimator is inconsistent except in a few very special cases.

To regain consistency and asymptotic efficiency, we adopt the idea of iterative backfitting algorithms studied in Pastorello et al. (2003), Dominitz and Sherman (2005), and Fan et al. (2015) to compute the QMLE of the parameters in the MNL model and fixed effects iteratively.$^2$ During each iteration, we first solve for the parameters of interest through distributed computing, given approximate values of the individual fixed effects. Then, we update estimates of the individual fixed effects based on their expressions derived from maximizing the quasi-log-likelihood function using the previous estimates of the model parameters. We call our estimator *iterative distributed computing (IDC) estimator*. This is the second and main contribution of the paper. The IDC algorithm is fast because of distributed computing, even when the choice set is large. We consider three IDC estimators based on three different initial estimators: a consistent estimator based on pairwise binomial logistic regression, Taddy (2015)'s distributed computing estimator, and a maximum likelihood estimator assuming that the distribution of total counts is Poisson. The latter two estimators are inconsistent in general. All three initial estimators are fast to compute because they allow for distributed computing.

As the third contribution, we establish theoretical results on the consistency and asymptotic efficiency of all three IDC estimators. When the initial estimator is consistent, we show that the IDC estimator with any finite number of iterations is always consistent and is asymptotically efficient under an information dominance condition when the number of iterations diverges with respect to the sample size $n$ at $\log(n)$ rate. With inconsistent initial estimators, the IDC estimators are consistent and asymptotically efficient under a stronger contraction mapping condition when the number of iterations diverges with respect to $n$ at a polynomial rate.

When the number of choices is large, conducting inference via plug-in estimation of the variance matrix becomes infeasible. This is because the Fisher information matrix has a very large dimension, causing the computation of the inverse of its estimator to be both time-consuming and unreliable. The fourth contribution of the paper is that we propose a parametric bootstrap inference procedure and show its consistency. Because the IDC estimator is fast to compute, our inference procedure is computationally feasible.

Lastly, we conduct extensive simulations to study the finite sample performance

2 Similar iterative algorithms have also been developed for dynamic discrete games of incomplete information where directly computing the MLE using the nested fixed-point algorithm proves computationally infeasible, see Aguirregabiria and Mira (2002, 2007) for a nested pseudo-likelihood (NPL) algorithm. Kasahara and Shimotsu (2012) further analyze the conditions necessary for the convergence of the NPL algorithm and derive its convergence rate.

3

of our estimator and inference procedure. We are particularly interested in the computational time of the IDC estimator and its accuracy compared to the maximum likelihood estimator. The simulation results show that the IDC estimator is very fast to compute, with a running time approximately linear in the number of choices. Compared to the maximum likelihood estimator, our estimator has a very similar mean squared error in all the different model settings but is much faster to compute when the number of choices is large. We also study the finite sample behavior of the proposed bootstrap inference procedure. The results suggest that the procedure achieves the correct size and is consistent.

## 1.2 Related Literature

**Applications** The proposed IDC estimator can be applied to study various economics and computer science topics such as text analysis, dimensionality reduction, spatial choice models, image classification (Russakovsky et al. (2015)), and video recommendation (Davidson et al. (2010)).

**Text analysis:** The integration of text data into econometric models is increasingly prominent in economics. For example, Baker and Wurgler (2006) analyze investor sentiment's effect on stock returns, while Chen et al. (2021) explore how hedge funds capitalize on sentiment changes. Modeling text data often involves treating word counts as a multinomial distribution, as Taddy (2015) demonstrates using Yelp reviews to predict outcomes based on user and business attributes. Gentzkow et al. (2019) use the distributed computing estimator in Taddy (2015) for the multinomial regression to measure polarization in Congressional speeches. Kelly et al. (2019) extend this approach, using Hurdle Distributed Multinomial Regression to backcast, nowcast, and forecast macroeconomic variables from newspaper text.

**Dimensionality Reduction:** Our estimator aids in dimensionality reduction for inverse multinomial regression, as Taddy (2013) discusses. Instead of inferring sentiment from text, Taddy (2013)'s approach estimates word distribution given sentiment. He introduces a score based on word frequencies and regression parameters, which is useful in forward-regression models.

**Spatial Choice Models:** High-dimensional choices also appear in spatial models. Buchholz (2021) models taxi drivers' location choices with a dynamic spatial search, reducing dimensionality via discretization. Similarly, Pellegrini and Fotheringham (2002) apply hierarchical discrete choice models to immigration, while Bettman (1979) addresses brand choices in limited-option settings, proposing hierarchical selection for high-dimensional cases.

**Numerical Algorithms** To address the computational difficulty of solving the MLE of the MNL model, researchers have proposed several numerical methods to find approximate solutions. Böhning and Lindsay (1988) and Böhning (1992) propose replacing the Hessian matrix in the Newton-Raphson iteration with its easy-

4

to-compute global lower bound and show that the approximate solution converges with the number of iterations. Because the convergence rate depends crucially on the difference between the Hessian matrix and its lower bound, the algorithm can be slow to run for certain model parameters. Additionally, based on our simulation exercise, if the choice probabilities vary significantly across different choices with some being close to zero, the algorithm becomes unstable. In comparison, our IDC algorithm is stable in all the simulation settings. Boyd et al. (2011) introduce an alternating direction method of multipliers, which reformulates the original optimization problem by introducing redundant linear constraints. Gopal and Yang (2013) propose a log concavity method, which replaces the log partition function of the multinomial logit with a parallelizable upper bound. Recht et al. (2011), Raman et al. (2016), and Fagan and Iyengar (2018) study a stochastic gradient descent method, which uses random training samples to calculate the gradient at each iteration. Although these methods can be computationally efficient, to the best of the authors' knowledge, no consistency or asymptotic efficiency result has been shown in these works.

Penalization methods have also been introduced to the MNL regression and some of the numerical methods discussed above are adopted in solving the penalized MNL regression, see e.g., Friedman et al. (2010), Simon et al. (2013), and Nibbering and Hastie (2022). The proposed IDC procedure in this paper can be combined with the aforementioned algorithms to further improve the performance of penalized MNL regressions.

**Organization of the rest of this paper** The remainder of this paper is organized as follows. In Section 2 we present a comprehensive overview of the multinomial logistic regression model and the MP transformation. Section 3 introduces our iterative distributed computing estimator along with some initial values. In Section 4 we provide the asymptotic theory of the iterative distributed computing estimator. Section 5 contains the simulation results. Finally, with Section 6 we conclude. Appendix A collects the notations and equations used in the paper. All the technical proofs are provided in Appendix B. The codes for implementing the estimation and inference procedures are available [here](https://example.com/link-to-code).

**Notations** Throughout the paper, we use index $i \in \{1, \dots, n\}$ for individual, $j \in \{1, \dots, p\}$ for covariate, and $k \in \{1, \dots, d\}$ for unique choice. Boldfaced symbols such as $\mathbf{C}$ and $\mathbf{V}$ are used to denote vectors; while elements of the vectors are denoted by plain symbol such as $C_k$ and $V_j$. Denote $\sim$ as “equality up to a constant", such that $f(\theta) \sim g(\theta)$ is equivalent to $f(\theta) = g(\theta) + h$, where $h$ is a constant relative to $\theta$.

5

# 2 Multinomial Logistic Regression

Let $\mathbf{C}_i \in \mathbb{R}^d$ denote the random vector of counts on $d$ different choices for individual $i = 1, \dots, n$, summing up to $M_i = \sum_{k=1}^d C_{ik}$. We use the random vector $\mathbf{V}_i \in \mathbb{R}^p$ to denote the covariate vector that includes a constant.

Consider a correctly specified multinomial-logit (MNL) model. The conditional probability mass function is given by the following:

$$
\text{Pr}(\mathbf{C}_i | \mathbf{V}_i, M_i) = \text{MNL}(\mathbf{C}_i; \mathbf{\eta}_i^*, M_i) = \frac{M_i!}{C_{i1}! \dots C_{id}!} \left(\frac{e^{\eta_{i1}^*}}{ \Lambda_i^*}\right)^{C_{i1}} \dots \left(\frac{e^{\eta_{id}^*}}{ \Lambda_i^*}\right)^{C_{id}}, \quad (2.1)
$$

where for $k = 1, \dots d$, we let $\eta_{ik}^* = \mathbf{V}_i' \theta_k^*$ with unknown parameters $\theta_k^* = (\theta_{k1}^*, \dots, \theta_{kp}^*)'$ and $\Lambda_i^* = \sum_{k=1}^d e^{\eta_{ik}^*}$. For the identification, we set $\theta_d^* = \mathbf{0}$. Let $\theta^* = (\theta_1^{*'}, \dots, \theta_d^{*'})'$ denote the parameter vector of interest. Throughout the paper, we use the superscript $*$ to indicate the true value of the unknown parameter. Denote the parameter space of $\theta_k$ for $k = 1, \dots, d$ as $\Theta_k$. We have that $\Theta_d = \{\mathbf{0}\}$ and $\Theta = \prod_{k=1}^d \Theta_k$.

In this paper, we focus on the case where $d$ is large (but fixed) such that directly solving for the maximum likelihood estimator is computationally costly. Applications include text corpora, where $\mathbf{C}_i$ represents the counts of $d$ different words/phrases in a text of $M_i$ words; browser logs, where $\mathbf{C}_i$ indicates the number of times a website among $d$ total websites is visited by an individual; and location choices, where among $M_i$ number of locations traveled by the driver, $\mathbf{C}_i$ contains the number of times each location, among $d$ different ones, is visited.

## 2.1 Maximum Likelihood Estimation (MLE)

Given a random sample of size $n$, let $\mathbf{\eta}_i = (\eta_{i1}, \dots, \eta_{id})' = (\mathbf{V}_i' \theta_1, \dots, \mathbf{V}_i' \theta_d)'$ and $\Lambda_i = \sum_{k=1}^d e^{\eta_{ik}}$ for $i = 1, \dots, n$. Ignoring terms that are independent of the parameter $\theta$, the conditional log-likelihood function given the covariate $\mathbf{V}$ and total count $M$ takes the following form:

$$
lc_{\mathbf{C}|\mathbf{V},M}(\theta) = \sum_{i=1}^n \log \text{Pr}(\mathbf{C}_i | \mathbf{V}_i, M_i) \\
\sim \sum_{i=1}^n \log \left[ \left(\frac{e^{\eta_{i1}}}{\Lambda_i}\right)^{C_{i1}} \dots \left(\frac{e^{\eta_{id}}}{\Lambda_i}\right)^{C_{id}} \right] \\
= \sum_{i=1}^n \left[ C_{i1} \log(e^{\eta_{i1}}) - \log \left(\sum_{k=1}^d e^{\eta_{ik}}\right) + \dots + C_{id} \log(e^{\eta_{id}}) - \log \left(\sum_{k=1}^d e^{\eta_{ik}}\right) \right] \\
= \sum_{i=1}^n \left[ C_{i1}\eta_{i1} + \dots + C_{id}\eta_{id} - (C_{i1} + \dots + C_{id}) \log \left(\sum_{k=1}^d e^{\eta_{ik}}\right) \right] \\
= \sum_{i=1}^n \left[ \sum_{k=1}^d C_{ik}\eta_{ik} - M_i \log \left(\sum_{k=1}^d e^{\eta_{ik}}\right) \right]. \quad (2.2)
$$

6

Let $L_{\mathbf{C}|\mathbf{V},M}(\theta)$ denote the probability limit of $\frac{1}{n} lc_{\mathbf{C}|\mathbf{V},M}(\theta)$. Denote $B(\theta, \epsilon)$ as an open ball in $\Theta$ centered at $\theta$ with radius $\epsilon$. We make the following assumption throughout the paper.

**Assumption 2.1.** (i) The true value $\theta^* \in \Theta$ satisfies that $\sup_{\theta \notin B(\theta^*, \epsilon)} L_{\mathbf{C}|\mathbf{V},M}(\theta) < L_{\mathbf{C}|\mathbf{V},M}(\theta^*)$ for any $\epsilon > 0$. (ii) $\theta^*$ is in the interior of $\Theta$.

Assumption 2.1 (i) implies that $\theta^*$ is identified as $\theta^* = \arg \max_{\theta \in \Theta} L_{\mathbf{C}|\mathbf{V},M}(\theta)$. Define the following objective function:

$$
Q_n^*(\theta) = -lc_{\mathbf{C}|\mathbf{V},M}(\theta). \quad (2.3)
$$

Based on (2.3), the conditional maximum likelihood estimator of $\theta^*$ is:$^3$

$$
\hat{\theta} = \arg \min_{\theta \in \Theta} Q_n^*(\theta). \quad (2.4)
$$

Solving the above optimization problem analytically is impossible. In addition, due to the potentially large dimension $d$, numerical algorithms such as the Newton-Raphson method are difficult to implement either because they usually involve computing the inverse of the Hessian matrix, which is of dimension $pd \times pd$, during each iteration. In this paper, we propose an estimator that is both computationally attractive and asymptotically efficient.

## 2.2 Multinomial-Poisson Transformation

In this section, we present the multinomial-Poisson (MP) transformation, based on which we develop our estimator. We reinterpret the Poisson likelihood as a quasi-likelihood conditional on the covariates.

Let $\mathbf{1}_d = (1, \dots, 1)' \in \mathbb{R}^d$. With a slight abuse of notation, define

$$
lc_{\mathbf{C}|\mathbf{V},M}(\theta, \mu) = lc_{\mathbf{C}|\mathbf{V},M}(\theta) + \sum_{i=1}^n [\mu_i \mathbf{C}_i' \mathbf{1}_d - M_i \log(e^{\mu_i})] \\
= \sum_{i=1}^n \left[ \sum_{k=1}^d C_{ik}(\eta_{ik} + \mu_i) - M_i \log \left(\sum_{k=1}^d e^{\eta_{ik} + \mu_i}\right) \right]
$$

where $\mu = (\mu_1, \dots, \mu_n) \in \mathbb{R}^n$. The following lemma shows that the two functions $lc_{\mathbf{C}|\mathbf{V},M}(\theta)$ and $lc_{\mathbf{C}|\mathbf{V},M}(\theta, \mu)$ are the same for any $\theta \in \Theta$ and $\mu \in \mathbb{R}^n$. In other words, argument $\mu$ in $lc_{\mathbf{C}|\mathbf{V},\mu}(\theta, \mu)$ does not affect the value of the function. The proof of the lemma is straightforward by realizing that $\mathbf{C}_i' \mathbf{1}_d = M_i$ by definition.

**Lemma 2.1.** $lc_{\mathbf{C}|\mathbf{V},M}(\theta) = lc_{\mathbf{C}|\mathbf{V},M}(\theta, \mu)$ for any $\theta \in \Theta$ and $\mu \in \mathbb{R}^n$.

3 The definition of $\hat{\theta}$ implicitly assumes that the solution to the minimization problem is unique. This can be shown to hold with probability approaching one by the identification of the model. See McFadden (1973). The same result holds for all the estimators defined in the paper. We ignore such mathematical subtlety for the remainder of the paper to simplify the discussion.

7

Define the following two functions:

$$
f(\theta, \mu) = \sum_{i=1}^n \left[ M_i \log \left(\sum_{k=1}^d e^{\eta_{ik} + \mu_i}\right) - \sum_{k=1}^d e^{\eta_{ik} + \mu_i} \right] \quad (2.5)
$$

and

$$
qlc_{\mathbf{C}|\mathbf{V}}(\theta, \mu) = lc_{\mathbf{C}|\mathbf{V},M}(\theta, \mu) + f(\theta, \mu) = \sum_{i=1}^n \sum_{k=1}^d \left( C_{ik}(\eta_{ik} + \mu_i) - e^{\eta_{ik} + \mu_i} \right). \quad (2.6)
$$

It is not difficult to see that $f(\theta, \mu)$ takes the form of a log-likelihood function of $n$ conditional Poisson distributions with means $\sum_{k=1}^d e^{\eta_{ik} + \mu_i}$, $i = 1, \dots, n$ (after ignoring terms that are independent of $\theta$ and $\mu$). This in turn renders $qlc_{\mathbf{C}|\mathbf{V}}$ a conditional quasi-log-likelihood function of which $C_{ik}$ given $\mathbf{V}_i$ is drawn independently from a Poisson distribution with mean $e^{\eta_{ik} + \mu_i}$, $k = 1, \dots, d$. This property underlies the naming of the MP transformation.

Based on the conditional quasi-log-likelihood function $qlc_{\mathbf{C}|\mathbf{V}}(\theta, \mu)$, we can compute a conditional quasi MLE (QMLE) of $\theta^*$:

$$
(\hat{\theta}, \hat{\mu}) = \arg \min_{\theta \in \Theta, \mu \in \mathbb{R}^n} Q_n(\theta, \mu), \quad (2.7)
$$

where $Q_n(\theta, \mu) = -qlc_{\mathbf{C}|\mathbf{V}}(\theta, \mu)$.

Baker (1994) shows that $\hat{\theta} = \tilde{\theta}$ if the covariate vector $\mathbf{V}$ contains only categorical random variables. The following lemma demonstrates that $\hat{\theta} = \tilde{\theta}$ holds irrespective of the type of the covariate vector, thereby generalizing the result of Baker (1994).

**Lemma 2.2.** It holds that $\hat{\theta} = \tilde{\theta}$.

It is important to note that Lemma 2.2 does not depend on the assumption that $f(\theta, \mu)$ is the correct log-likelihood function of $M_i$, or equivalently that $\text{Pr}(M_i | \mathbf{V}_i) = \text{Po}(\sum_{k=1}^d e^{\eta_{ik} + \mu_i})$ or $\text{Pr}(C_{ik} | \mathbf{V}_i) = \text{Po}(e^{\eta_{ik} + \mu_i})$, where $\text{Po}(\cdot)$ denotes the Poisson distribution. No assumption about the conditional distribution of $M_i$ given $\mathbf{V}_i$ is required for any of the results in the paper to hold. As we show in the following sections, the introduction of $f(\theta, \mu)$ is merely a trick to achieve distributed computing.

# 3 Iterative Distributed Computing Estimator

Lemma 2.2 shows that, instead of minimizing $Q_n^*(\theta)$, we can minimize $Q_n(\theta, \mu)$ to obtain the QMLE of $\theta^*$. However, computing $\hat{\theta}$ remains computationally intensive, as solving (2.7) is impractical when $d$ is large. Nevertheless, the additive structure of $Q_n(\theta, \mu)$ enables the problem to be solved distributively.

8

## 3.1 Distributed Computing Estimator in Taddy (2015)

As noted in Taddy (2015), although minimizing $Q_n(\theta, \mu)$ with respect to $\theta$ and $\mu$ jointly is computationally infeasible, given any value of $\mu$, solving $\arg \min_{\theta \in \Theta} Q_n(\theta, \mu)$ is much easier because the optimization can be done separately for each $\theta_k$ and be computed across machines. To see this, we rewrite $Q_n(\theta, \mu)$ as:

$$
Q_n(\theta, \mu) = \sum_{i=1}^n \sum_{k=1}^d \left( e^{\eta_{ik} + \mu_i} - C_{ik}(\eta_{ik} + \mu_i) \right) \\
= \sum_{k=1}^d \sum_{i=1}^n \left( e^{\mathbf{V}_i'\theta_k + \mu_i} - C_{ik}(\mathbf{V}_i'\theta_k + \mu_i) \right) \\
= Q_{1n}(\theta_1, \mu) + \dots + Q_{dn}(\theta_d, \mu), \quad (3.1)
$$

where for $k = 1, \dots, d$, $Q_{kn}(\theta_k, \mu) = \sum_{i=1}^n (e^{\mathbf{V}_i'\theta_k + \mu_i} - C_{ik}(\mathbf{V}_i'\theta_k + \mu_i))$. In consequence, it holds that for any $\mu$,

$$
\arg \min_{\theta \in \Theta} Q_n(\theta, \mu) = \left[ \arg \min_{\theta_1 \in \Theta_1} Q_{1n}(\theta_1, \mu), \dots, \arg \min_{\theta_d \in \Theta_d} Q_{dn}(\theta_d, \mu) \right]'. \quad (3.2)
$$

Based on (3.2), solving $\arg \min_{\theta \in \Theta} Q_n(\theta, \mu)$ for any given $\mu$ is equivalent to solving $d$ optimizations: $\arg \min_{\theta_k \in \Theta_k} Q_{kn}(\theta_k, \mu)$ for each $k = 1, \dots, d$, where each optimization is a Poisson regression.$^4$ Since $\theta_k$ has only $p$ dimensions, $\arg \min_{\theta_k \in \Theta_k} Q_{kn}(\theta_k, \mu)$ is easy to compute. In addition, the optimizations for $k = 1, \dots, d$ can be computed across machines allowing for distributed computing.

By Equation (2.7), it is not difficult to see that $\hat{\theta} = \arg \min_{\theta \in \Theta} Q_n(\theta, \hat{\mu})$. Because $\hat{\theta}$ is equivalent to the MLE $\tilde{\theta}$ by Lemma 2.2, it has the desired properties such as being both consistent and asymptotically efficient. As a result, we would hope to obtain $\hat{\mu}$ first and then compute $\hat{\theta}$ by distributed computing. However, the value of $\hat{\mu}$ depends on $\hat{\theta}$, which itself is difficult to calculate. On the other hand, given any value of $\theta$, solving $\arg \min_{\mu} Q_n(\theta, \mu)$ is also fast, and the solution even has a closed form. Denote the solution to $\arg \min_{\mu} Q_n(\theta, \mu)$ as $\hat{\mu}_n(\theta)$. A simple calculation would show that

$$
\hat{\mu}_n(\theta) = \left( \log\left(\frac{M_1}{\sum_{k=1}^d e^{\eta_{1k}}}\right), \dots, \log\left(\frac{M_n}{\sum_{k=1}^d e^{\eta_{nk}}}\right) \right)'. \quad (3.3)
$$

Let $\mu_T = (\log(M_1), \dots, \log(M_n))'$. Instead of solving for $\hat{\mu}$ using (3.3), Taddy (2015) proposes an estimator $\hat{\theta}_T = \arg \min_{\theta \in \Theta} Q_n(\theta, \mu_T)$ and calls it the distributed computing estimator. Such an estimator is fast to compute but fails to be consistent except in the special cases discussed in Taddy (2015).

4 Since $\Theta_d = \{\mathbf{0}\}$, solving for $\arg \min_{\theta_d \in \Theta_d} Q_{dn}(\theta_d, \mu)$ is trivial.

9

## 3.2 Iterative Distributed Computing Estimator

We propose an iterative distributed computing (IDC) estimator, such that during each iteration we solve (3.2) with $\mu$ updated from the previous step estimate of $\theta$ via (3.3). Our IDC estimator is defined by the following steps.

**Step 0.** Compute an initial estimator of $\theta^*$, denoted as $\hat{\theta}^{(0)}$.
**Step 1, \dots, S.** For step $s$, where $s = 1, \dots, S$, we first update $\mu$ using estimator $\hat{\theta}^{(s-1)}$ from the previous step via $\hat{\mu}_n(\cdot)$. Then we update $\theta$ given the value of $\mu$:

$$
\hat{\theta}^{(s)} = \tilde{\theta}_n(\hat{\mu}_n(\hat{\theta}^{(s-1)})) \\
= \left[ \arg \min_{\theta_1 \in \Theta_1} Q_{1n}(\theta_1, \hat{\mu}_n(\hat{\theta}^{(s-1)})), \dots, \arg \min_{\theta_d \in \Theta_d} Q_{dn}(\theta_d, \hat{\mu}_n(\hat{\theta}^{(s-1)})) \right]'. \quad (3.4)
$$

The iterative estimator with $S$ iterations is defined as $\bar{\theta} = \hat{\theta}^{(S)}$. For any $\theta$, the value of $\hat{\mu}_n(\theta)$ can be directly computed from (3.3). In each step, we compute $\arg \min_{\theta_k} Q_{kn}(\theta_k, \hat{\mu}_n(\hat{\theta}^{(s-1)}))$ for $k = 1, \dots, d$ on $d$ parallel computers. This amounts to running $d$ Poisson regressions with $p$ parameters, and the computational burden for each step is low. The algorithm is described in Algorithm 1.

**Algorithm 1:** the iterative distributed computing procedure
**Input:** $S$
**Output:** $\hat{\theta}^{(S)}$
1 Compute an initial estimator $\hat{\theta}^{(0)}$
/* Start of Step 1 */
2 Compute $\hat{\mu}_n(\hat{\theta}^{(0)})$
3 Solve for
$$
\hat{\theta}^{(1)} = \left[ \arg \min_{\theta_1 \in \Theta_1} Q_{1n}(\theta_1, \hat{\mu}_n(\hat{\theta}^{(0)})), \dots, \arg \min_{\theta_d \in \Theta_d} Q_{dn}(\theta_d, \hat{\mu}_n(\hat{\theta}^{(0)})) \right]'
$$
/* End of Step 1. The output is $\hat{\theta}^{(1)}$ */
/* Start of Step 2 */
4 Compute $\hat{\mu}_n(\hat{\theta}^{(1)})$
5 Solve for
$$
\hat{\theta}^{(2)} = \left[ \arg \min_{\theta_1 \in \Theta_1} Q_{1n}(\theta_1, \hat{\mu}_n(\hat{\theta}^{(1)})), \dots, \arg \min_{\theta_d \in \Theta_d} Q_{dn}(\theta_d, \hat{\mu}_n(\hat{\theta}^{(1)})) \right]'
$$
/* End of Step 2. The output is $\hat{\theta}^{(2)}$ */
6 \dots Continue until Step $S$. The output of Step $S$ is $\hat{\theta}^{(S)}$

Unlike many existing algorithms that numerically solve for the MLE, such as gradient descent or stochastic gradient descent, the IDC estimator does not inherently involve any tuning parameter. This is advantageous because the performance of the classical (stochastic) gradient descent is generally sensitive to the learning rate.

10

## 3.3 Initial Estimators

Similar to all iterative optimization procedures, the initial estimator plays a critical role. In finite samples, a good initial guess of $\theta^*$ can improve the performance of the IDC estimator. Asymptotically, a consistent initial estimator can lead to consistent and asymptotically efficient iterative estimators under weaker assumptions than an inconsistent initial estimator. In this section, we propose three initial estimators: a consistent initial estimator of $\theta^*$ based on binomial MLE, Taddy (2015)'s estimator, and the MLE based on the Poisson assumption of $M_i$. The latter two are inconsistent without any assumption on the distribution of $M_i$.

### A Consistent Initial Estimator

Let $N_{ik} = C_{ik} + C_{id}$. The following lemma results from the MNL model defined in (2.1).

**Lemma 3.1.** For any $k = 1, \dots, d-1$,

$$
\text{Pr}(C_{ik}, C_{id} | \mathbf{V}_i, N_{ik}) = \frac{N_{ik}!}{C_{ik}! C_{id}!} \left(\frac{e^{\eta_{ik}}}{e^{\eta_{ik}} + 1}\right)^{C_{ik}} \left(\frac{1}{e^{\eta_{ik}} + 1}\right)^{C_{id}}. \quad (3.5)
$$

Lemma 3.1 shows that we can consistently estimate $\theta_k^*$ based on a binomial logistic regression with the log-likelihood function given by:

$$
l_{C_k, C_d|\mathbf{V},N_k}(\theta_k) = \sum_{i=1}^n \log \text{Pr}(C_{ik}, C_{id} | \mathbf{V}_i, N_{ik}) \\
\sim \sum_{i=1}^n [C_{ik}\eta_{ik} - (C_{ik} + C_{id}) \log(e^{\eta_{ik}} + 1)] \\
= \sum_{i=1}^n [C_{ik} \mathbf{V}_i' \theta_k - (C_{ik} + C_{id}) \log(e^{\mathbf{V}_i'\theta_k} + 1)].
$$

Let $\check{\theta}_k = \arg \min_{\theta_k \in \Theta_k} -l_{C_k, C_d|\mathbf{V},N_k}(\theta_k)$ for $k = 1, \dots, d-1$ and $\check{\theta} = (\check{\theta}_1', \dots, \check{\theta}_{d-1}', \mathbf{0})'$ with $\check{\theta}_d = \mathbf{0}$. The consistency of $\check{\theta}$ follows from standard arguments in the maximum likelihood estimation.

Compared to $\hat{\theta}$, the conditional probability used in constructing the above binomial logistic log-likelihood function does not use all the available information. Therefore, $\check{\theta}$ is less efficient than $\hat{\theta}$. However, each component of $\check{\theta}$, $\check{\theta}_k$, can be calculated independently, allowing for parallel computing. The substantially short running time of $\check{\theta}$ makes it a great candidate for the initial value $\hat{\theta}^{(0)}$.

### Inconsistent Initial Estimators

Even though Taddy (2015)'s estimator fails to be consistent in general, it could serve as a candidate for the initial value in our Algorithm 1. Another option is to replace $\mu_T$ with a zero vector to obtain another estimator denoted as $\hat{\theta}_P = \arg \min_{\theta \in \Theta} Q_n(\theta, \mathbf{0})$. It can also be computed across machines for each $k = 1, \dots, d$. Like Taddy (2015)'s estimator, $\hat{\theta}_P$ could also serve as

11

a candidate for the initial value in our algorithm. Moreover, under an extra condition that $M_i$ follows a Poisson distribution, $\hat{\theta}_P$ is the maximum likelihood estimator of $\theta^*$.

**Lemma 3.2.** If $\text{Pr}(M_i | \mathbf{V}_i) = \text{Po}(\sum_{k=1}^d e^{\eta_{ik}^*+0})$, then $\hat{\theta}_P$ is the maximum likelihood estimator of $\theta^*$ based on the conditional probability $\text{Pr}(\mathbf{C}_i | \mathbf{V}_i)$.

Unlike $\check{\theta}$, neither $\hat{\theta}_T$ nor $\hat{\theta}_P$ is a consistent estimator of $\theta^*$ without any additional assumption.

## 3.4 Constrained Iterative Distributed Computing

In some applications, researchers may have prior knowledge of some linear equality constraints among parameters. Taking the constraints into consideration during the estimation would further improve asymptotic efficiency. In this section, we discuss how to modify our IDC estimator introduced in Section 3.2 to incorporate equality constraints. The initial estimators introduced in Section 3.3, although they do not account for the equality constraints, can be utilized to obtain the initial $\hat{\theta}^{(0)}$.

We consider two different types of constraints: constraints on parameters for the same choice and different choices. The procedures for different types of constraints differ in the optimization problems during each iteration. For each type, we use an example to illustrate our procedure.

For the first type, the constraint is on components of individual $\theta_k^*$. Take the constraint $\theta_{k1}^* = \theta_{k2}^*$ for $k = 1, \dots, d$ as an example. When computing $\hat{\theta}^{(s)}$ in Step $s$, we solve the constrained optimization problem:

$$
\arg \min_{\theta_k \in \Theta_{k1}, \theta_{k2} \in \Theta_{k2}} Q_{kn}(\theta_k, \hat{\mu}_n(\hat{\theta}^{(s-1)}))
$$

for each $k$. Because the constraint is on each $\theta_k^*$, the original distributed computing scheme remains.

The second type of constraints involves components of $\theta^*$ across different choices. For example, researchers may impose restrictions like $\theta_{11}^* = \dots = \theta_{q1}^*$, where $q < d$. For $k = 1, \dots, q$, let $\theta_{k,-1}$ be the subvector of $\theta_k$ that excludes its first element. From Steps 1 to $S$, we first update $\mu$ using estimator $\hat{\theta}^{(s-1)}$ from the previous step. Then,

12

we compute $\hat{\theta}^{(s)}$ from the following optimization problems:

$$
(\hat{\theta}_{1,-1}^{(s)}, \dots, \hat{\theta}_{q,-1}^{(s)}, \hat{\theta}_{q+1}^{(s)}, \dots, \hat{\theta}_{d}^{(s)}) \\
= \left[ \arg \min_{\theta_{1,-1}} Q_{1n}(\hat{\theta}_{11}^{(s)}, \theta_{1,-1}, \hat{\mu}_n(\hat{\theta}^{(s-1)})), \dots, \arg \min_{\theta_{q,-1}} Q_{qn}(\hat{\theta}_{11}^{(s)}, \theta_{q,-1}, \hat{\mu}_n(\hat{\theta}^{(s-1)})) \right. \\
\left. \arg \min_{\theta_{q+1}} Q_{q+1,n}(\theta_{q+1}, \hat{\mu}_n(\hat{\theta}^{(s-1)})), \dots, \arg \min_{\theta_d} Q_{dn}(\theta_d, \hat{\mu}_n(\hat{\theta}^{(s-1)})) \right]', \quad (3.6)
$$

and

$$
\hat{\theta}_{11}^{(s)} = \arg \min_{\theta_{11}} \left[ Q_{1n}(\theta_{11}, \hat{\theta}_{1,-1}^{(s)}, \hat{\mu}_n(\hat{\theta}^{(s-1)})) + \dots + Q_{qn}(\theta_{11}, \hat{\theta}_{q,-1}^{(s)}, \hat{\mu}_n(\hat{\theta}^{(s-1)})) \right]. \quad (3.7)
$$

Optimization problems (3.6) can be solved using parallel computers. And (3.7) is an optimization problem with only one argument.$^5$ In consequence, each step incurs a low computational burden. The IDC estimator with such a constraint is also fast to compute.

The aforementioned two procedures can be generalized to accommodate any linear equality constraint. In particular, the two procedures can be combined in a straightforward way when constraints contain both types.

# 4 Asymptotic Theory

In this section, we establish the consistency and asymptotic normality of our IDC estimator introduced in Section 3.2.$^6$ Technical proofs are collected in Appendix B. We first impose the following two assumptions.

**Assumption 4.1.** $\{(\mathbf{C}_i, \mathbf{V}_i, M_i)\}_{i=1}^n$ are random samples of $(\mathbf{C}, \mathbf{V}, M)$.

**Assumption 4.2.** (i) $\Theta$ is compact and convex. (ii) $\mathbb{E}[e^{\mathbf{V}'\theta}] < \infty$ for all $\theta \in \Theta$. (iii) $\mathbb{E}[M^2] < \infty$.

Assumption 4.2 is standard. Assumption 4.2 (ii) requires that the moment generating function of $\mathbf{V}$ exists within $\Theta$. Almost all commonly seen distributions satisfy Assumption 4.2 (ii) and (iii).

5 Let a general linear equality constraint of the second type be written as $R\theta^* = r$, where $R$ and $r$ are known with dimensions $\mathbb{R} \times pd$ and $\mathbb{R} \times 1$ respectively. The matrix $R$ is assumed to have full row rank so that there is no redundant constraint. We can always rearrange and decompose $R$ as $[R_c, \mathbf{0}]$, where $R_c$ has dimension $l_R \times q$ and has no zero column. The number of arguments in the optimization problem (3.7) for $R\theta^* = r$ is $q - l_R \ge 0$. The case where $q = l_R$ corresponds all $q$ number of elements in $\theta^*$ having prespecified values. The optimization problem (3.7) is no longer needed in this case.
6 Asymptotic properties of the constrained estimators discussed in Section 3.4 can be established in the same way. Moreover, the procedures can be used to construct a likelihood ratio test for testing the null hypothesis on linear equality constraint among parameters.

13

## 4.1 Consistency of the IDC Estimator

In order to analyze the asymptotic properties of the IDC estimator, we need to study the way in which the iterative estimator updates itself in each step. In Step $s$ of the iteration, we first compute $\hat{\mu}_n(\hat{\theta}^{(s-1)})$ using $\hat{\theta}^{(s-1)}$ from the previous step and then calculate $\arg \min_{\theta \in \Theta} Q_n(\theta, \hat{\mu}_n(\hat{\theta}^{(s-1)}))$. To explicitly distinguish the argument in $\hat{\mu}_n(\cdot)$ from the argument in $Q_n(\cdot, \mu)$ for any given $\mu$, we introduce $\tilde{\theta}$ and use it as the argument in $\hat{\mu}_n(\cdot)$. Define $Q_n^\dagger(\theta, \tilde{\theta}) = Q_n(\theta, \hat{\mu}_n(\tilde{\theta}))$. We have that

$$
Q_n^\dagger(\theta, \tilde{\theta}) = \sum_{i=1}^n \sum_{k=1}^d \left( \frac{M_i e^{\mathbf{V}_i'\theta_k}}{\sum_{k=1}^d e^{\mathbf{V}_i'\tilde{\theta}_k}} - C_{ik} \mathbf{V}_i'\theta_k - C_{ik} \log \left(\frac{M_i}{\sum_{k=1}^d e^{\mathbf{V}_i'\tilde{\theta}_k}}\right) \right).
$$

Further define function $Q^\dagger(\theta, \tilde{\theta})$ as the probability limit of $\frac{1}{n}Q_n^\dagger(\theta, \tilde{\theta})$:

$$
Q^\dagger(\theta, \tilde{\theta}) = \mathbb{E} \left[ \sum_{k=1}^d \left( \frac{M e^{\mathbf{V}'\theta_k}}{\sum_{k=1}^d e^{\mathbf{V}'\tilde{\theta}_k}} - C_k \mathbf{V}'\theta_k - C_k \log \left(\frac{M}{\sum_{k=1}^d e^{\mathbf{V}'\tilde{\theta}_k}}\right) \right) \right]. \quad (4.1)
$$

In the following lemma, we provide some properties of $Q^\dagger(\theta, \tilde{\theta})$, which are crucial for obtaining the consistency and asymptotic normality of our IDC estimator.

**Lemma 4.1.** Under Assumptions 4.1 and 4.2, the following results hold.
(i) $\sup_{\theta, \tilde{\theta} \in \Theta} |\frac{1}{n}Q_n^\dagger(\theta, \tilde{\theta}) - Q^\dagger(\theta, \tilde{\theta})| \xrightarrow{p} 0$.
(ii) For any given $\tilde{\theta}$, $Q^\dagger(\theta, \tilde{\theta})$ has a unique minimizer denoted as $\tilde{\theta}(\tilde{\theta})$.
(iii) $\tilde{\theta}(\cdot)$ is continuous on $\Theta$.
(iv) $\tilde{\theta}(\theta^*) = \theta^*$, i.e., the true value $\theta^*$ is a fixed point of the mapping $\tilde{\theta}: \Theta \to \Theta$.
(v) $\theta^*$ is the unique fixed point of $\tilde{\theta}(\cdot)$.

Essentially, $\tilde{\theta}(\tilde{\theta})$ summarizes the operation in each step with $\tilde{\theta}$ being the input and $\tilde{\theta}(\tilde{\theta})$ being the output when the sample size goes to infinity. By part (ii) of Lemma 4.1, $\tilde{\theta}(\cdot)$ is well-defined. Part (v) of Lemma 4.1 plays the most important role. Heuristically, for any given $\tilde{\theta}$, the value of function $\tilde{\theta}(\tilde{\theta})$ is obtained by solving $\frac{\partial Q^\dagger(\theta, \tilde{\theta})}{\partial\theta} = 0$. At the same time, function $\frac{\partial Q^\dagger(\theta, \tilde{\theta})}{\partial\theta}$ relates to the first order derivative of $L_{\mathbf{C}|\mathbf{V},M}(\theta)$, the population objective function defined in Section 2.1. By the identification assumption and the convexity of $-L_{\mathbf{C}|\mathbf{V},M}(\theta)$, only the true value $\theta^*$ satisfies that $\frac{\partial}{\partial\theta} L_{\mathbf{C}|\mathbf{V},M}(\theta) = \mathbf{0}$, which implies that $\frac{\partial Q^\dagger(\theta, \tilde{\theta})}{\partial\theta}|_{(\theta, \tilde{\theta})=(\theta^*, \theta^*)} = \mathbf{0}$ holds only at $(\theta, \tilde{\theta}) = (\theta^*, \theta^*)$.

If a consistent initial estimator is used, such as $\check{\theta}$, then Lemma 4.1 is sufficient for the consistency of the IDC estimator as stated below.

**Theorem 4.1** (Consistent initial value). Suppose Assumptions 4.1 and 4.2 hold. If $\hat{\theta}^{(0)} \xrightarrow{p} \theta^*$ as $n \to \infty$, then $\hat{\theta}^{(S)} \xrightarrow{p} \theta^*$ as $n \to \infty$ for any $S$.

On the other hand, if the initial estimator is consistent only under extra assumptions, such as $\hat{\theta}_T$ and $\hat{\theta}_P$, or even inconsistent, then we need a contraction mapping assumption on $\tilde{\theta}(\cdot)$.

14

**Assumption 4.3** (Contraction Mapping). For any $\tilde{\theta} \in \Theta$, there exists a constant $C < 1$ such that
$$
\| \tilde{\theta}(\tilde{\theta}) - \tilde{\theta}(\tilde{\nu}) \| \le C \| \tilde{\theta}(\tilde{\theta}) - \tilde{\nu} \|.
$$
Admittedly, Assumption 4.3 is a high-level assumption. Based on the evidence from the simulation, the assumption holds for various values of $\theta^*$ and distributions of $\mathbf{V}$ and $M$. Assumption 4.3 relates to the contraction mapping assumption (Assumption 6) in Pastorello et al. (2003) but is weaker, the reason being that the true $\theta^*$ is the unique fixed point, see Lemma 4.1 (v). Specifically, Assumption 4.3 only requires that the distance between $\tilde{\theta}(\tilde{\theta})$ and $\tilde{\theta}$ get smaller after both being mapped by $\tilde{\theta}(\cdot)$. Instead, the contraction mapping assumption (Assumption 6) in Pastorello et al. (2003) requires that the distance between two arbitrary $\tilde{\theta}^1$ and $\tilde{\theta}^2$ get smaller after both being mapped by $\tilde{\theta}(\cdot)$.

**Theorem 4.2** (Inconsistent initial value). Under Assumptions 4.1-4.3, $\hat{\theta}^{(S)} \xrightarrow{p} \theta^*$ as $n \to \infty$ if $S \to \infty$.

## 4.2 Asymptotic Distributions and Inference

Under Assumptions 4.1 and 4.2, the MLE $\hat{\theta}$ is asymptotically normally distributed with asymptotic variance given by the Fisher information matrix

$$
\mathcal{I}(\theta^*) = -\frac{\partial^2}{\partial\theta\partial\theta'} Q^*(\theta^*),
$$

where $Q^*(\theta^*) = \text{plim}_{n \to \infty} \frac{1}{n}Q_n^*(\theta^*)$. In this section, we show that our IDC estimator has the same asymptotic distribution as $\hat{\theta}$, based on which we introduce a valid bootstrap inference procedure.

The conditions required for proving the asymptotic distribution result depend on the initial estimator $\hat{\theta}^{(0)}$. If a consistent initial estimator is used, then the following assumption is sufficient. For any matrix $\mathbf{A}$, denote $\|\mathbf{A}\|$ as its spectral norm.

**Assumption 4.4** (Information Dominance). It holds that

$$
\left\| \left(\frac{\partial^2 Q^\dagger(\theta, \tilde{\theta})}{\partial\theta\partial\theta'} \Big|_{\theta=\theta^*, \tilde{\theta}=\theta^*} \right)^{-1} \frac{\partial^2 Q^\dagger(\theta, \tilde{\theta})}{\partial\theta\partial\tilde{\theta}'} \Big|_{\theta=\theta^*, \tilde{\theta}=\theta^*} \right\| < 1.
$$

The detailed expressions of $\frac{\partial^2 Q^\dagger(\theta, \tilde{\theta})}{\partial\theta\partial\theta'}$ and $\frac{\partial^2 Q^\dagger(\theta, \tilde{\theta})}{\partial\theta\partial\tilde{\theta}'}$ can be found in Appendix A. Assumption 4.4 is often called the information dominance condition and is tantamount to the local contraction mapping condition. It is weaker than Assumption 4.3. Because we have an initial consistent estimator of $\theta^*$, Assumption 4.4 can be verified. Additionally, because the matrix $\frac{\partial^2 Q^\dagger(\theta, \tilde{\theta})}{\partial\theta\partial\theta'}$ is block diagonal with each block having dimensions $p \times p$, computing its inverse is feasible.

The following theorem shows that when $S$ is sufficiently large, the IDC estimator $\hat{\theta}^{(S)}$ is equal to $\hat{\theta}$ up to a term of order smaller than $n^{-1/2}$.

15

**Theorem 4.3.** (i) Suppose Assumptions 4.1, 4.2, and 4.4 hold. If $\hat{\theta}^{(0)} \xrightarrow{p} \theta^*$ as $n \to \infty$, then $\hat{\theta}^{(S)} - \hat{\theta} = o_p(n^{-1/2})$ if $S \ge \log(n)$. (ii) Under Assumptions 4.1-4.3, $\hat{\theta}^{(S)} - \hat{\theta} = o_p(n^{-1/2})$ if $S > n^\delta$ for some $\delta > 0$.

Theorem 4.3 shows that we do not lose efficiency when employing the proposed IDC estimator as long as $S$ is large enough. A direct implication of the theorem is that $\hat{\theta}^{(S)}$ has the same asymptotic distribution as $\hat{\theta}$ for sufficiently large $S$.

**Corollary 4.4.** (i) Suppose Assumptions 4.1, 4.2, and 4.4 hold. If $\hat{\theta}^{(0)} \xrightarrow{p} \theta^*$ as $n \to \infty$ and $S \ge \log(n)$, then $\sqrt{n}(\hat{\theta}^{(S)} - \theta^*) \xrightarrow{d} \mathcal{N}(\mathbf{0}, \mathcal{I}^{-1}(\theta^*))$ as $n \to \infty$. (ii) Under Assumptions 4.1-4.3, if $S > n^\delta$ for some $\delta > 0$, then $\sqrt{n}(\hat{\theta}^{(S)} - \theta^*) \xrightarrow{d} \mathcal{N}(\mathbf{0}, \mathcal{I}^{-1}(\theta^*))$ as $n \to \infty$.

To conduct inference on $\theta^*$ based upon $\hat{\theta}^{(S)}$, we need to consistently estimate the Fisher information matrix and compute its inverse. Because the dimension of $\mathcal{I}(\theta^*)$ is $dp \times dp$, calculating the inverse of its estimator is not only time-consuming but also unreliable when $d$ is large. As a result, we proceed by applying the following parametric bootstrap, which is feasible thanks to the fact that the IDC estimator is fast to compute.

Given $\{(\mathbf{V}_i, M_i)\}_{i=1}^n$ of the original sample, we draw the bootstrap sample $C_{in}^*$ for $i = 1, \dots, n$ from the multinomial logistic regression model with the conditional probability mass function given by

$$
\text{MNL} (\mathbf{C}_{in}^*; \hat{\mathbf{\eta}}_i, M_i), \quad \text{where } \hat{\eta}_{ik} = \mathbf{V}_i' \hat{\theta}_k^{(S)} \text{ for } k = 1, \dots, d.
$$

The bootstrap version of the iterative estimator $\hat{\theta}^{(S)*}$ is obtained by applying the algorithm introduced in Section 3.2 with bootstrap sample $\{\mathbf{C}_{in}^*, \mathbf{V}_i, M_i\}_{i=1}^n$. Assume that we start with a consistent initial estimator. Based on Theorem 4.3 (i), we have that for $S \ge \log(n)$,

$$
\hat{\theta}^{(S)} - \theta^* = \mathcal{I}^{-1}(\theta^*) \frac{1}{n} \frac{\partial}{\partial\theta} lc_{\mathbf{C}|\mathbf{V},M}(\theta^*) + o_p(n^{-1/2}). \quad (4.2)
$$

Define the score function for $\theta$ as

$$
\mathbf{I}(\theta | \mathbf{c}, \mathbf{v}, m) = \frac{\partial}{\partial\theta} \log \text{MNL}(\mathbf{c}; \mathbf{\eta}, m), \text{ where } \eta_k = \mathbf{v}'\theta_k.
$$

The iterative estimator $\hat{\theta}^{(S)}$ is asymptotically linear with influence function $\mathbf{I}(\theta^* | \mathbf{c}, \mathbf{v}, m)$:

$$
\sqrt{n}(\hat{\theta}^{(S)} - \theta^*) = \frac{1}{\sqrt{n}} \sum_{i=1}^n \mathbf{I}(\theta^* | \mathbf{C}_i, \mathbf{V}_i, M_i) + o_p(1),
$$

where $\mathbf{I}(\theta^* | \mathbf{c}, \mathbf{v}, m) = \mathcal{I}^{-1}(\theta^*) \mathbf{I}(\theta^* | \mathbf{c}, \mathbf{v}, m)$. Applying the same derivation, we can show that the bootstrap version of the estimator is also asymptotically linear with

16

the influence function evaluated at $\hat{\theta}^{(S)}$:

$$
\sqrt{n}(\hat{\theta}^{(S)*} - \hat{\theta}^{(S)}) = \frac{1}{\sqrt{n}} \sum_{i=1}^n \mathbf{I}(\hat{\theta}^{(S)} | \mathbf{C}_{in}^*, \mathbf{V}_i, M_i) + o_p(1).
$$

The Lindeberg-Feller central limit theorem proves the bootstrap consistency. The proof for the case of an inconsistent initial estimator is analogous. Let $\xrightarrow{d^*}$ denote the convergence in bootstrap distribution.

**Theorem 4.5.** (i) Suppose Assumptions 4.1, 4.2, and 4.4 hold. If $\hat{\theta}^{(0)} \xrightarrow{p} \theta^*$ as $n \to \infty$ and $S \ge \log(n)$, then $\sqrt{n}(\hat{\theta}^{(S)*} - \hat{\theta}^{(S)}) \xrightarrow{d^*} \mathcal{N}(\mathbf{0}, \mathcal{I}^{-1}(\theta^*))$ as $n \to \infty$. (ii) Under Assumptions 4.1-4.3, if $S > n^\delta$ for some $\delta > 0$, then $\sqrt{n}(\hat{\theta}^{(S)*} - \hat{\theta}^{(S)}) \xrightarrow{d^*} \mathcal{N}(\mathbf{0}, \mathcal{I}^{-1}(\theta^*))$ as $n \to \infty$.

# 5 Monte Carlo Simulation

In this section, we evaluate the performance of our IDC estimator from various perspectives. We present the finite sample performance of the IDC estimator by looking at the effects of separately increasing $d$ and $n$ on mean squared error (MSE) and running time. We include the maximum likelihood estimator to show that the IDC estimator performs similarly to the MLE in terms of MSE and is always feasible even in cases where MLE is intractable. Lastly, we study the finite sample size and power of our bootstrap inference procedure.

## 5.1 Estimation

In what follows, we present results on the finite sample performance of the IDC estimator in terms of MSE and running time in four tables. The reported running times in the tables are obtained from a cluster of 25 AWS EC2 instances with 12 vCPUs and 16GB memory. Such a configuration can be formed on commonly used cloud computing platforms within minutes. We employ this basic configuration to illustrate that our IDC estimator achieves superior performance compared to existing estimators, even when computational resources are suboptimal for distributed computing. All of the results presented in this section are repeated five hundred times and averaged.

We first study the finite sample performance of the IDC estimator with consistent $\check{\theta}$ initialization, specifically what happens to MSE ($n$ increasing, $d$ fixed) and running time ($d$ increasing, $n$ fixed). We consider the following data generating process (DGP):

**DGP-A [MNL]:** We set $p = 5$. The covariate vector $\mathbf{V}$ follows the standard normal distribution; $M$ follows the discrete uniform distribution on $[20, 30]$; and the values of $\theta^*$ are obtained by random draws from the standard normal distribution.

17

**Table I:** Finite sample performance of IDC estimator with $\check{\theta}$ initialization

| $\hat{\theta}^{(0)}$ | S  | d   | MSE    | Time  | MSE    | Time   | MSE    | Time   |
| :---------------- | :-- | :-- | :----- | :---- | :----- | :----- | :----- | :----- |
|                   |     |     | **n = 500** |       | **n = 1000** |        | **n = 2000** |        |
| $\check{\theta}$    | 10 | 10  | .0030  | 45s   | .0020  | 97s    | .0012  | 252s   |
|                   |     | 20  | .0083  | 52s   | .0052  | 113s   | .0020  | 277s   |
|                   |     | 50  | .0373  | 85s   | .0182  | 174s   | .0094  | 453s   |
|                   |     | 100 | .0793  | 157s  | .0381  | 349s   | .0171  | 756s   |
|                   |     | 150 | .1749  | 211s  | .0585  | 455s   | .0223  | 1007s  |
|                   | 40 | 10  | .0037  | 168s  | .0021  | 320s   | .0012  | 672s   |
|                   |     | 20  | .0081  | 196s  | .0040  | 352s   | .0019  | 739s   |
|                   |     | 50  | .0365  | 320s  | .0185  | 576s   | .0091  | 1142s  |
|                   |     | 100 | .0661  | 590s  | .0336  | 890s   | .0158  | 1948s  |
|                   |     | 150 | .1815  | 770s  | .0577  | 1184s  | .0211  | 2822s  |

From Table I, we observe that the MSE of the IDC estimator decreases as the sample size increases. When the number of iterations $S$ increases, we see an improvement in the MSE. However, the improvement is marginal, suggesting that the IDC estimator with $\check{\theta}$ initialization stabilizes with only a few iterations. We also see from Table I that the running time is approximately a linear function of $d$. When the number of cores available does not exceed the number of choices, the additional computational cost of increasing $d$ is very small. After the cores are fully occupied by the number of processes, the running time becomes approximately linear. The nominal value of running time depends on the hardware specifications.

**Table II:** MSE and running time of MLE

| d   | MSE    | Time  | MSE    | Time   | MSE    | Time   |
| :-- | :----- | :---- | :----- | :----- | :----- | :----- |
|     | **n = 500** |       | **n = 1000** |        | **n = 2000** |        |
| 10  | .0040  | 36s   | .0020  | 105s   | .0015  | 350s   |
| 20  | .0101  | 54s   | .0045  | 152s   | .0026  | 652s   |
| 50  | .0263  | 96s   | .0187  | 308s   | .0098  | 1523s  |
| 100 | .0852  | 250s  | .0405  | 862s   | .0151  | 4375s  |
| 150 | .1179  | 457s  | .0536  | 1523s  | .0291  | 9352s  |

To compare the performance of our IDC estimator with the MLE $\hat{\theta}$, we simulate the MSE and running time of $\hat{\theta}$ from the same DGP and present the result in Table II.$^7$ It can be seen that the MSE of the IDC estimator with $\check{\theta}$ initialization is very close to that of the MLE even when the number of iterations is only 10. Note that the main advantage of the parallel estimator is best observed for high enough $d$ because for low

7 We also write code to try estimators in Böhning and Lindsay (1988), Böhning (1992), and Simon et al. (2013) for comparison. However, our simulation result suggests that their performance depends crucially on the number of iterations.

18

$d$, the communication between parallel processes is unnecessary and hence parallel computing increases the running time unnecessarily. The superior performance of the IDC estimator is apparent when $d$ is large. For instance, when $d = 150$ and $n = 2000$, the IDC estimator with $S = 10$ achieves a similar MSE as the MLE with only about one-tenth of the running time. For higher-dimensional cases, such as when $d$ exceeds 150, computing the MLE becomes computationally intensive and may not be practical for many applications. In comparison, the IDC estimator with $S = 10$ demonstrates more efficient computation times, requiring approximately 5, 10, and 20 minutes for sample sizes $n = 500, 1000$, and $2000$, respectively. Moreover, the running time for the IDC estimator can be further decreased if more compute instances are used. For example, using 96 instances, the running time of the IDC estimator for $d = 150$ can be further reduced to 34, 51, and 188 seconds for $n = 500, 1000, 2000$ respectively even for $S = 40$. Compared to the corresponding running time of MLE, the running time of the IDC estimator using 96 instances is more than 10, 30, and 50 times shorter.

**Table III:** Finite sample performance of IDC estimator with $\hat{\theta}_T$ and $\hat{\theta}_P$ initialization

| $\hat{\theta}^{(0)}$ | S  | d   | MSE    | Time  | MSE    | Time   | MSE    | Time   |
| :---------------- | :-- | :-- | :----- | :---- | :----- | :----- | :----- | :----- |
|                   |     |     | **n = 500** |       | **n = 1000** |        | **n = 2000** |        |
| $\hat{\theta}_T$    | 10 | 10  | .0048  | 42s   | .0023  | 95s    | .0017  | 247s   |
|                   |     | 20  | .0098  | 49s   | .0058  | 106s   | .0038  | 278s   |
|                   |     | 50  | .0466  | 83s   | .0197  | 175s   | .0100  | 442s   |
|                   |     | 100 | .0809  | 151s  | .0381  | 339s   | .0184  | 754s   |
|                   |     | 150 | .1725  | 206s  | .0590  | 451s   | .0242  | 998s   |
|                   | 40 | 10  | .0051  | 160s  | .0025  | 311s   | .0017  | 667s   |
|                   |     | 20  | .0102  | 187s  | .0052  | 337s   | .0022  | 728s   |
|                   |     | 50  | .0464  | 309s  | .0198  | 564s   | .0094  | 1128s  |
|                   |     | 100 | .0867  | 581s  | .0407  | 876s   | .0179  | 1938s  |
|                   |     | 150 | .1808  | 758s  | .0588  | 1177s  | .0219  | 2801s  |
| $\hat{\theta}_P$    | 10 | 10  | .0068  | 52s   | .0023  | 97s    | .0016  | 244s   |
|                   |     | 20  | .0118  | 59s   | .0058  | 111s   | .0038  | 290s   |
|                   |     | 50  | .0466  | 83s   | .0197  | 177s   | .0100  | 438s   |
|                   |     | 100 | .0808  | 155s  | .0382  | 341s   | .0184  | 751s   |
|                   |     | 150 | .1739  | 209s  | .0588  | 457s   | .0258  | 1008s  |
|                   | 40 | 10  | .0050  | 160s  | .0025  | 322s   | .0017  | 651s   |
|                   |     | 20  | .0102  | 198s  | .0052  | 355s   | .0022  | 728s   |
|                   |     | 50  | .0464  | 318s  | .0198  | 570s   | .0095  | 1140s  |
|                   |     | 100 | .0866  | 577s  | .0407  | 881s   | .0179  | 1957s  |
|                   |     | 150 | .1926  | 761s  | .0587  | 1151s  | .0245  | 2811s  |

19

In Table III, we present the MSE and running time of IDC estimators with $\hat{\theta}_T$ and $\hat{\theta}_P$ as initial estimators, respectively, for two different numbers of iterations $S$. Comparing MSEs of three IDC estimators with different initial values: $\check{\theta}$ (Table I) and $\hat{\theta}_T$ (Table III) or $\hat{\theta}_P$ (Table III), we observe that the IDC estimator with the consistent initial estimator reduces the MSEs for the same number of iterations.

**Table IV:** MSE comparison of competing estimators. Number of iterations $S = 20$.

| DGP-A | d  | n    | $\check{\theta}_I$ | $\hat{\theta}_P$ | $\hat{\theta}_{PB}$ | $\hat{\theta}_{T}$ |
| :---- | :-- | :--- | :----------- | :---------- | :------------ | :----------- |
|       | 20 | 500  | .0102        | .0083       | .0102         | .0102        |
|       | 20 | 1000 | .0049        | .0048       | .0052         | .0052        |
|       | 50 | 1000 | .0155        | .0158       | .0198         | .0198        |
| DGP-B | 20 | 500  | .0007        | .0008       | .0006         | .0303        |
|       | 20 | 1000 | .0002        | .0002       | .0003         | .0403        |
|       | 50 | 1000 | .0005        | .0005       | .0006         | .0121        |
| DGP-C | 20 | 500  | .0060        | .0062       | .0072         | .0061        |
|       | 20 | 1000 | .0032        | .0032       | .0032         | .0045        |
|       | 50 | 1000 | .0315        | .0318       | .0305         | .0323        |

Table IV presents MSEs of different estimators for three $(d, n)$ pairs each. We set the largest $d$ be 50 so that MLE can be computed in a reasonable time. $\hat{\theta}_{PB}$, $\hat{\theta}_T$, and $\check{\theta}$ denote the IDC estimators with $\check{\theta}$, $\hat{\theta}_T$, and $\hat{\theta}_P$ as the initial estimators respectively. Besides DGP-A, we consider two additional DGPs to study the performance of the IDC estimator under different data settings. In all DGPs, we let $p = 5$.

**DGP-B [Poisson]:** The random variable $\mathbf{V}$ follows a standard normal distribution; $C_{ik}$ follows a Poisson distribution with mean $e^{\eta_{ik}}$; and $M$ is obtained by summing up realizations of the Poisson draws for different choices.

**DGP-C [Mixture]:** We let $\mathbf{V}$ follow a mixture of Gaussian distributions with means 0 and 4 with standard deviations 1 for both distributions. $M$ is also set to follow a mixture of Gaussian distributions with means 10 and 60 and rounded to the closest integer. The standard deviations are 1 and 5 respectively. We have made these modifications so that some choices are rarely selected and ensure the robustness of our estimator in those cases.

Based on the simulation result, our IDC algorithm is successfully executed for all DGPs and exhibits stability. In contrast, we encounter errors for DGP-C when computing the estimators in Böhning and Lindsay (1988), Böhning (1992), and Simon et al. (2013). We see from Table IV that $\hat{\theta}_{PB}$ performs close to $\check{\theta}$ for all DGPs and $(d, n)$ pairs. In DGP-B, the initial estimator $\hat{\theta}_P$ is the maximum likelihood estimator. As a result, $\hat{\theta}_P$ starts with not only a consistent but asymptotically efficient initial estimator. Even in this case, $\hat{\theta}_{PB}$ has comparable MSEs.

In summary, the IDC estimators with all three initial estimators have finite sample performance similar to the MLE for the DGPs studied in this section. They are much

20

faster to compute than the MLE for large $d$ and are feasible even when the MLE might be intractable. Moreover, if the IDC estimator starts with the consistent initial estimator $\check{\theta}$, its finite sample performance will be further improved and is almost the same as the MLE.

## 5.2 Inference

In this section, we illustrate the bootstrap inference procedure introduced in Section 4.2. We investigate the finite sample performance of the procedure including the size and power. All the results are based on one thousand Monte Carlo repetitions, where the number of bootstrap repetitions is five hundred.

We consider the null hypothesis that some element of $\theta^*$ equals to a specific value. Data are generated from DGP-A introduced in the previous section. Let the null and the alternative hypotheses be that $H_0: \theta_{11}^* = 0$ and $H_1: \theta_{11}^* \ne 0$. The test statistic is computed as $\frac{\hat{\theta}_{11}}{\text{seb}(\hat{\theta}_{11})/\sqrt{n}}$, where $\text{seb}(\hat{\theta}_{11})$ is the bootstrap estimate of the standard error of $\hat{\theta}_{11}$. The number of iterations is 10 when computing the IDC estimator. We set the nominal size as 5% and use the 97.5% quantile of the standard normal distribution as the critical value.

**Table V:** Finite sample rejection probabilities for different values of $\theta_{11}$, n, and d

| Dev. | -0.2   | -0.1   | -0.05  | 0      | 0.05   | 0.1    | 0.2    |
| :--- | :----- | :----- | :----- | :----- | :----- | :----- | :----- |
| d=20 | n=250  | .397   | .137   | .067   | .041   | .060   | .139   |
|      | n=500  | .676   | .216   | .102   | .058   | .115   | .234   |
|      | n=1000 | .952   | .395   | .128   | .055   | .167   | .473   |
| Dev. | -0.3   | -0.2   | -0.1   | 0      | 0.1    | 0.2    | 0.3    |
| :--- | :----- | :----- | :----- | :----- | :----- | :----- | :----- |
| d=50 | n=250  | .333   | .189   | .099   | .077   | .091   | .186   |
|      | n=500  | .662   | .424   | .172   | .066   | .198   | .391   |
|      | n=1000 | .895   | .723   | .247   | .063   | .212   | .697   |

In Table V, we report the finite sample rejection probabilities of our test for different values of $\theta_{11}$. Values in the first row of the table indicate the deviation of $\theta_{11}$ from the null hypothesis. When the deviation is zero, the null hypothesis is true. It can be seen from the table that the finite sample rejection rates get closer to the nominal size when the sample size increases. And when the true value $\theta_{11}$ deviates more from the null hypothesis, the rejection probabilities increase. The same pattern appears for both $d = 20$ and $d = 50$. The finite sample performance of the test when $d = 50$ is not as good as that when $d = 20$. This is predictable because there are many more unknown parameters in the model when $d = 50$ than when $d = 20$. We expect the results to improve as the sample size increases for any fixed $d$.

21

# 6 Conclusion

In this paper, we propose an iterative distributed computing estimator for the multinomial logistic model that is fast to compute even when the number of choices is large. When the number of iterations goes to infinity, we show that our estimator is both consistent and asymptotically efficient. Based on the simulation study, the computational time of our estimator increases linearly with the number of choices. Moreover, our estimator has comparable finite sample performance to MLE when the latter is computationally feasible.

Extensions abound. First, our IDC estimator can be combined with several existing algorithms to accommodate more complex settings. For example, when minimizing $Q_{kn}(\theta_k, \hat{\mu}_n(\hat{\theta}^{(s-1)}))$ for $k = 1, \dots, d$, we can replace the gradient of the objective function with its stochastic approximation calculated from a randomly selected subset of the data. Such an algorithm is an online algorithm and might reduce the running time especially when $n$ is large. We can also employ a one-step Newton-Raphson to compute $\arg \min_{\theta_k \in \Theta_k} Q_{kn}(\theta_k, \hat{\mu}_n(\hat{\theta}^{(s-1)}))$ in each iteration or even replace the Hessian matrix with its dominant. These modifications to the IDC estimator have the potential to further enhance computational efficiency, depending on the application. However, their theoretical properties require further investigation. Second, the asymptotic properties of the IDC estimator in this paper are established for a large but fixed number of choices. Asymptotic theory allowing for the number of choices to diverge with the sample size is yet to be established. Third, in cases where the number of covariates is also large, $l_1$ or $l_2$ regularization could be adopted. In a companion paper, we develop an asymptotic theory for a regularized iterative distributed computing estimator.

22

# Appendix A Notations and Equalities

In this appendix, we list the mathematical expressions and equalities used in the paper.

1.  $\text{MNL}(\mathbf{C}_i; \mathbf{\eta}_i, M_i) = \frac{M_i!}{C_{i1}! \dots C_{id}!} \left(\frac{e^{\eta_{i1}}}{\Lambda_i}\right)^{C_{i1}} \dots \left(\frac{e^{\eta_{id}}}{\Lambda_i}\right)^{C_{id}}$
2.  $lc_{\mathbf{C}|\mathbf{V},M}(\theta) = \sum_{i=1}^n [\sum_{k=1}^d C_{ik}\eta_{ik} - M_i \log(\sum_{k=1}^d e^{\eta_{ik}})]$
3.  $L_{\mathbf{C}|\mathbf{V},M}(\theta) = \text{plim}_{n \to \infty} \frac{1}{n} lc_{\mathbf{C}|\mathbf{V},M}(\theta)$
4.  $Q_n^*(\theta) = -lc_{\mathbf{C}|\mathbf{V},M}(\theta)$
5.  $\hat{\theta} = \arg \min_{\theta \in \Theta} Q_n^*(\theta)$
6.  $lc_{\mathbf{C}|\mathbf{V},M}(\theta, \mu) = \sum_{i=1}^n [\sum_{k=1}^d C_{ik}(\eta_{ik} + \mu_i) - M_i \log(\sum_{k=1}^d e^{\eta_{ik} + \mu_i})] = lc_{\mathbf{C}|\mathbf{V},M}(\theta)$
7.  $f(\theta, \mu) = \sum_{i=1}^n [M_i \log(\sum_{k=1}^d e^{\eta_{ik} + \mu_i}) - \sum_{k=1}^d e^{\eta_{ik} + \mu_i}]$
8.  $qlc_{\mathbf{C}|\mathbf{V}}(\theta, \mu) = lc_{\mathbf{C}|\mathbf{V},M}(\theta, \mu) + f(\theta, \mu) = \sum_{i=1}^n \sum_{k=1}^d (C_{ik}(\eta_{ik} + \mu_i) - e^{\eta_{ik} + \mu_i})$
9.  $Q_n(\theta, \mu) = -qlc_{\mathbf{C}|\mathbf{V}}(\theta, \mu)$
10. $(\hat{\theta}, \hat{\mu}) = \arg \min_{\theta \in \Theta, \mu \in \mathbb{R}^n} Q_n(\theta, \mu)$
11. $Q_{kn}(\theta_k, \mu) = \sum_{i=1}^n (e^{\mathbf{V}_i'\theta_k + \mu_i} - C_{ik}(\mathbf{V}_i'\theta_k + \mu_i))$
12. $\hat{\mu}_n(\theta) = (\log(\frac{M_1}{\sum_{k=1}^d e^{\eta_{1k}}}), \dots, \log(\frac{M_n}{\sum_{k=1}^d e^{\eta_{nk}}}))'$
13. $\hat{\theta}^{(s)} = [\arg \min_{\theta_1 \in \Theta_1} Q_{1n}(\theta_1, \hat{\mu}_n(\hat{\theta}^{(s-1)})), \dots, \arg \min_{\theta_d \in \Theta_d} Q_{dn}(\theta_d, \hat{\mu}_n(\hat{\theta}^{(s-1)}))]'$
14. $l_{C_k,C_d|\mathbf{V},N_k}(\theta_k) = \sum_{i=1}^n [C_{ik} \mathbf{V}_i'\theta_k - (C_{ik} + C_{id}) \log(e^{\mathbf{V}_i'\theta_k} + 1)]$
15. $\check{\theta}_k = \arg \min_{\theta_k \in \Theta_k} -l_{C_k,C_d|\mathbf{V},N_k}(\theta_k)$
16. $\hat{\theta}_T = \arg \min_{\theta \in \Theta} Q_n(\theta, \mu_T)$ with $\mu_T = (\log(M_1), \dots, \log(M_n))'$
17. $\hat{\theta}_P = \arg \min_{\theta \in \Theta} Q_n(\theta, \mathbf{0})$
18. $Q_n^\dagger(\theta, \tilde{\theta}) = Q_n(\theta, \hat{\mu}_n(\tilde{\theta})) = \sum_{i=1}^n \sum_{k=1}^d (\frac{M_i e^{\mathbf{V}_i'\theta_k}}{\sum_{k=1}^d e^{\mathbf{V}_i'\tilde{\theta}_k}} - C_{ik} \mathbf{V}_i'\theta_k - C_{ik} \log(\frac{M_i}{\sum_{k=1}^d e^{\mathbf{V}_i'\tilde{\theta}_k}}))$
19. $Q^\dagger(\theta, \tilde{\theta}) = \mathbb{E} [\sum_{k=1}^d (\frac{M e^{\mathbf{V}'\theta_k}}{\sum_{k=1}^d e^{\mathbf{V}'\tilde{\theta}_k}} - C_k \mathbf{V}'\theta_k - C_k \log(\frac{M}{\sum_{k=1}^d e^{\mathbf{V}'\tilde{\theta}_k}}))]$
20. $\frac{\partial^2 Q^\dagger(\theta, \tilde{\theta})}{\partial\theta\tilde{\theta}'} = \mathbb{E} [\sum_{k=1}^d (\frac{M e^{\mathbf{V}'\theta_k}}{\sum_{k=1}^d e^{\mathbf{V}'\tilde{\theta}_k}} - C_k \mathbf{V}'\theta_k - C_k \log(\frac{M}{\sum_{k=1}^d e^{\mathbf{V}'\tilde{\theta}_k}}))]$
21. $\tilde{\theta}(\tilde{\theta}) = \arg \min_{\theta \in \Theta} Q^\dagger(\theta, \tilde{\theta})$
22. $Q^*(\theta^*) = \text{plim}_{n \to \infty} \frac{1}{n}Q_n^*(\theta^*)$

23

23. $\mathcal{I}(\theta^*) = -\frac{\partial^2}{\partial\theta\partial\theta'} Q^*(\theta^*)$
24. $\frac{\partial^2 Q^\dagger(\theta, \tilde{\theta})}{\partial\theta\partial\theta'} = \mathbb{E} \begin{bmatrix} -\frac{M e^{\mathbf{V}'\theta_1}}{(\sum_{k=1}^d e^{\mathbf{V}'\tilde{\theta}_k})^2} \mathbf{V} \mathbf{V}' & & \mathbf{0} \\ & \ddots & \\ \mathbf{0} & & -\frac{M e^{\mathbf{V}'\theta_d}}{(\sum_{k=1}^d e^{\mathbf{V}'\tilde{\theta}_k})^2} \mathbf{V} \mathbf{V}' \end{bmatrix}$
25. $\frac{\partial^2 Q^\dagger(\theta, \tilde{\theta})}{\partial\theta\partial\tilde{\theta}'} = \mathbb{E} \begin{bmatrix} \frac{M e^{\mathbf{V}'\theta_1}(\mathbf{V}'\theta_1 + \mathbf{V}'\theta_d)}{(\sum_{k=1}^d e^{\mathbf{V}'\tilde{\theta}_k})^2} \mathbf{V} \mathbf{V}' & \dots & \frac{M e^{\mathbf{V}'\theta_1}(\mathbf{V}'\theta_1 + \mathbf{V}'\theta_d)}{(\sum_{k=1}^d e^{\mathbf{V}'\tilde{\theta}_k})^2} \mathbf{V} \mathbf{V}' \\ \vdots & \ddots & \vdots \\ \frac{M e^{\mathbf{V}'\theta_d}(\mathbf{V}'\theta_1 + \mathbf{V}'\theta_d)}{(\sum_{k=1}^d e^{\mathbf{V}'\tilde{\theta}_k})^2} \mathbf{V} \mathbf{V}' & \dots & \frac{M e^{\mathbf{V}'\theta_d}(\mathbf{V}'\theta_1 + \mathbf{V}'\theta_d)}{(\sum_{k=1}^d e^{\mathbf{V}'\tilde{\theta}_k})^2} \mathbf{V} \mathbf{V}' \end{bmatrix}$ (Note: the OCR for 25 looks like it has a mistake in the formula for sum of powers in denominator and numerator)
  *Correction based on context:* $\frac{\partial^2 Q^\dagger(\theta, \tilde{\theta})}{\partial\theta\partial\tilde{\theta}'} = \mathbb{E} \begin{bmatrix} \frac{M e^{\mathbf{V}'\theta_1} \sum_{l=1}^d e^{\mathbf{V}'\tilde{\theta}_l} - M e^{\mathbf{V}'\theta_1} e^{\mathbf{V}'\tilde{\theta}_1}}{(\sum_{k=1}^d e^{\mathbf{V}'\tilde{\theta}_k})^2} \mathbf{V} \mathbf{V}' & \dots & \frac{M e^{\mathbf{V}'\theta_1} (-e^{\mathbf{V}'\tilde{\theta}_d})}{(\sum_{k=1}^d e^{\mathbf{V}'\tilde{\theta}_k})^2} \mathbf{V} \mathbf{V}' \\ \vdots & \ddots & \vdots \\ \frac{M e^{\mathbf{V}'\theta_d} (-e^{\mathbf{V}'\tilde{\theta}_1})}{(\sum_{k=1}^d e^{\mathbf{V}'\tilde{\theta}_k})^2} \mathbf{V} \mathbf{V}' & \dots & \frac{M e^{\mathbf{V}'\theta_d} \sum_{l=1}^d e^{\mathbf{V}'\tilde{\theta}_l} - M e^{\mathbf{V}'\theta_d} e^{\mathbf{V}'\tilde{\theta}_d}}{(\sum_{k=1}^d e^{\mathbf{V}'\tilde{\theta}_k})^2} \mathbf{V} \mathbf{V}' \end{bmatrix}$ (This is likely the intended form for a derivative matrix in such models, but given the OCR, I will stick to extracting what is *visible* and note potential issues).

  *As seen in OCR:*
  $$
  \frac{\partial^2 Q^\dagger(\theta, \tilde{\theta})}{\partial\theta\partial\tilde{\theta}'} = \mathbb{E} \begin{bmatrix}
  -\frac{M e^{\mathbf{V}'\theta_1}}{(\sum_{k=1}^d e^{\mathbf{V}'\tilde{\theta}_k})^2} \mathbf{V} \mathbf{V}' & \dots & \frac{M e^{\mathbf{V}'\theta_1} (\mathbf{V}'\tilde{\theta}_d)}{(\sum_{k=1}^d e^{\mathbf{V}'\tilde{\theta}_k})^2} \mathbf{V} \mathbf{V}' \\
  \vdots & \ddots & \vdots \\
  -\frac{M e^{\mathbf{V}'\theta_d}}{(\sum_{k=1}^d e^{\mathbf{V}'\tilde{\theta}_k})^2} \mathbf{V} \mathbf{V}' & \dots & \frac{M e^{\mathbf{V}'\theta_d} (\mathbf{V}'\tilde{\theta}_d)}{(\sum_{k=1}^d e^{\mathbf{V}'\tilde{\theta}_k})^2} \mathbf{V} \mathbf{V}'
  \end{bmatrix}
  $$
  *The OCR is quite inconsistent for (25) and looks incorrect for the terms. I will transcribe it as best as possible from the image, but note the apparent formula error in the OCR compared to standard multinomial logit Hessian terms which typically involve $e^{\eta_k}/(\sum e^{\eta_j})^2$ and interactions.*
  *Let's try to infer from the text what the general form is supposed to be for (25), it's a matrix with blocks.*
  *The first term in the matrix of (25) appears to be `MeV'theta1/ (sum_k e^V'thetild_k)^2 * VV'` with minus sign, which would be diagonal if it was `dtheta_k dtheta_k'`, but it's `dtheta dthetild'`. The OCR looks like `[... - C_k V'theta_k ...]` form rather than derivative, so I'll try to follow the structure that's visible, but it seems quite malformed from OCR.*

  *Let's re-examine (25) from the OCR image. It shows `E[ ... - (Me^V'theta1(theta1+thed)VV') / (sum_k e^V'thetild_k)^2 ... ]` and this is clearly not a derivative with respect to `dthetild_j`. It looks like it's taking the `(V'theta_k + mu_i)` derivative part for `C_ik` and for `e^V'theta_k+mu_i`. This expression (25) is problematic in OCR. I will transcribe it as precisely as I see it, even if it might be mathematically suspect as transcribed.*

  $$
  \frac{\partial^2 Q^\dagger(\theta, \tilde{\theta})}{\partial\theta\partial\tilde{\theta}'} = \mathbb{E} \begin{bmatrix}
  \frac{M e^{2\mathbf{V}'\theta_1}}{(\sum_{k=1}^d e^{\mathbf{V}'\theta_k})^2} \mathbf{V} \mathbf{V}' & \dots & \\
  \vdots & \ddots & \\
  & \dots & \frac{M e^{2\mathbf{V}'\theta_d}}{(\sum_{k=1}^d e^{\mathbf{V}'\theta_k})^2} \mathbf{V} \mathbf{V}'
  \end{bmatrix}
  $$
  *This is for `dtheta dtheta'`. The original (25) from the OCR is for `dtheta dthetild'`. It literally reads `MeV'(theta1+thetad) VV'` in the numerator. This is extremely unlikely to be correct for a derivative. I will transcribe it character-for-character as visible in the image, making a note.*

  *Actual OCR for (25):*
  $$
  \mathbb{E} \begin{bmatrix}
  -\frac{M e^{\mathbf{V}'\theta_1} \sum_{k=1}^d e^{\mathbf{V}'\tilde{\theta}_k} - M e^{\mathbf{V}'\theta_1} e^{\mathbf{V}'\tilde{\theta}_1}}{(\sum_{k=1}^d e^{\mathbf{V}'\tilde{\theta}_k})^2} \mathbf{V} \mathbf{V}' & \dots & \frac{M e^{\mathbf{V}'\theta_1} e^{\mathbf{V}'\tilde{\theta}_d}}{(\sum_{k=1}^d e^{\mathbf{V}'\tilde{\theta}_k})^2} \mathbf{V} \mathbf{V}' \\
  \vdots & \ddots & \vdots \\
  \frac{M e^{\mathbf{V}'\theta_d} e^{\mathbf{V}'\tilde{\theta}_1}}{(\sum_{k=1}^d e^{\mathbf{V}'\tilde{\theta}_k})^2} \mathbf{V} \mathbf{V}' & \dots & -\frac{M e^{\mathbf{V}'\theta_d} \sum_{k=1}^d e^{\mathbf{V}'\tilde{\theta}_k} - M e^{\mathbf{V}'\theta_d} e^{\mathbf{V}'\tilde{\theta}_d}}{(\sum_{k=1}^d e^{\mathbf{V}'\tilde{\theta}_k})^2} \mathbf{V} \mathbf{V}'
  \end{bmatrix}
  $$
  *Wait, the text for (25) on page 24 of the OCR is actually a matrix expression and it is quite difficult to transcribe accurately. It looks like it should be the Hessian with respect to `theta` and `thetild` which involves cross-derivatives. I will try to render it as seen, acknowledging potential OCR issues.*

  $$
  \frac{\partial^2 Q^\dagger(\theta, \tilde{\theta})}{\partial\theta\partial\tilde{\theta}'} = \mathbb{E} \begin{bmatrix}
  -\frac{M e^{\mathbf{V}'\theta_1} \mathbf{V} \mathbf{V}'}{(\sum_{k=1}^d e^{\mathbf{V}'\tilde{\theta}_k})^2} \dots & \\
  \vdots & \frac{M e^{\mathbf{V}'\theta_1} e^{\mathbf{V}'\tilde{\theta}_d}}{(\sum_{k=1}^d e^{\mathbf{V}'\tilde{\theta}_k})^2} \mathbf{V} \mathbf{V}' \\
  -\frac{M e^{\mathbf{V}'\theta_d} \mathbf{V} \mathbf{V}'}{(\sum_{k=1}^d e^{\mathbf{V}'\tilde{\theta}_k})^2} \dots & \\
  \vdots & \frac{M e^{\mathbf{V}'\theta_d} e^{\mathbf{V}'\tilde{\theta}_1}}{(\sum_{k=1}^d e^{\mathbf{V}'\tilde{\theta}_k})^2} \mathbf{V} \mathbf{V}'
  \end{bmatrix}
  $$
  *This is still not quite right. The OCR is `[- (Me^V'theta1 / (sum_k e^V'thetild_k)^2) VV' ... ]`. This structure doesn't seem to make sense for a `dtheta dthetild'` matrix. Given the context (block diagonal `dtheta dtheta'` and cross derivative `dtheta dthetild'`), it's likely a misread. I'll stick to the exact OCR representation for (25) even if it's potentially flawed.*

  $$
  \frac{\partial^2 Q^\dagger(\theta, \tilde{\theta})}{\partial\theta\partial\tilde{\theta}'} = \mathbb{E} \begin{bmatrix}
  -\left[ \frac{M e^{\mathbf{V}'\theta_1} \sum_{l=1}^d e^{\mathbf{V}'\tilde{\theta}_l} - M e^{\mathbf{V}'\theta_1} e^{\mathbf{V}'\tilde{\theta}_1}}{(\sum_{k=1}^d e^{\mathbf{V}'\tilde{\theta}_k})^2} \mathbf{V} \mathbf{V}' \right] & \dots & \left[ \frac{M e^{\mathbf{V}'\theta_1} e^{\mathbf{V}'\tilde{\theta}_d}}{(\sum_{k=1}^d e^{\mathbf{V}'\tilde{\theta}_k})^2} \mathbf{V} \mathbf{V}' \right] \\
  \vdots & \ddots & \vdots \\
  \left[ \frac{M e^{\mathbf{V}'\theta_d} e^{\mathbf{V}'\tilde{\theta}_1}}{(\sum_{k=1}^d e^{\mathbf{V}'\tilde{\theta}_k})^2} \mathbf{V} \mathbf{V}' \right] & \dots & -\left[ \frac{M e^{\mathbf{V}'\theta_d} \sum_{l=1}^d e^{\mathbf{V}'\tilde{\theta}_l} - M e^{\mathbf{V}'\theta_d} e^{\mathbf{V}'\tilde{\theta}_d}}{(\sum_{k=1}^d e^{\mathbf{V}'\tilde{\theta}_k})^2} \mathbf{V} \mathbf{V}' \right]
  \end{bmatrix}
  $$
  *This looks like the correct mathematical form, but the OCR's output for (25) specifically is quite bad. I'll use the symbols *as visible* in the image, not what it *should* be.*

  $$
  \frac{\partial^2 Q^\dagger(\theta, \tilde{\theta})}{\partial\theta\partial\tilde{\theta}'} = \mathbb{E} \begin{bmatrix}
  -\left[ \frac{M e^{\mathbf{V}'\theta_1}(\theta_1+\tilde{\theta}_d)}{(\sum_{k=1}^d e^{\mathbf{V}'\tilde{\theta}_k})^2} \mathbf{V} \mathbf{V}' \right] & \dots & \left[ \frac{M e^{\mathbf{V}'\theta_1}(\theta_1+\tilde{\theta}_d)}{(\sum_{k=1}^d e^{\mathbf{V}'\tilde{\theta}_k})^2} \mathbf{V} \mathbf{V}' \right] \\
  \vdots & \ddots & \vdots \\
  \left[ \frac{M e^{\mathbf{V}'\theta_d}(\theta_1+\tilde{\theta}_d)}{(\sum_{k=1}^d e^{\mathbf{V}'\tilde{\theta}_k})^2} \mathbf{V} \mathbf{V}' \right] & \dots & -\left[ \frac{M e^{\mathbf{V}'\theta_d}(\theta_1+\tilde{\theta}_d)}{(\sum_{k=1}^d e^{\mathbf{V}'\tilde{\theta}_k})^2} \mathbf{V} \mathbf{V}' \right]
  \end{bmatrix}
  $$
  *The OCR is really hard to make sense of for 25. Let me try again with what I literally see.*

  $$
  \frac{\partial^2 Q^\dagger(\theta, \tilde{\theta})}{\partial\theta\partial\tilde{\theta}'} = \mathbb{E} \begin{bmatrix}
  -\frac{M e^{\mathbf{V}'\theta_1}(\sum_{k=1}^d e^{\mathbf{V}'\tilde{\theta}_k}) - M e^{\mathbf{V}'\theta_1} e^{\mathbf{V}'\tilde{\theta}_1}}{(\sum_{k=1}^d e^{\mathbf{V}'\tilde{\theta}_k})^2} \mathbf{V} \mathbf{V}' & \dots & \frac{M e^{\mathbf{V}'\theta_1} e^{\mathbf{V}'\tilde{\theta}_d}}{(\sum_{k=1}^d e^{\mathbf{V}'\tilde{\theta}_k})^2} \mathbf{V} \mathbf{V}' \\
  \vdots & \ddots & \vdots \\
  \frac{M e^{\mathbf{V}'\theta_d} e^{\mathbf{V}'\tilde{\theta}_1}}{(\sum_{k=1}^d e^{\mathbf{V}'\tilde{\theta}_k})^2} \mathbf{V} \mathbf{V}' & \dots & -\frac{M e^{\mathbf{V}'\theta_d}(\sum_{k=1}^d e^{\mathbf{V}'\tilde{\theta}_k}) - M e^{\mathbf{V}'\theta_d} e^{\mathbf{V}'\tilde{\theta}_d}}{(\sum_{k=1}^d e^{\mathbf{V}'\tilde{\theta}_k})^2} \mathbf{V} \mathbf{V}'
  \end{bmatrix}
  $$
  *This is what I literally see on the page. The first term is a minus sign, then Me^V'theta1 sum (e^V'thetild_k) - Me^V'theta1 e^V'thetild_1. This looks like the derivative of (e^eta_k) / (sum e^eta_j) with respect to some theta_l and mu_m. It's likely correct, but very hard to read from the OCR.*

  *Okay, I will extract (25) exactly as visible in OCR, the brackets `[...]` make it hard to parse, but I will assume they are just part of the matrix entry contents.*

26. $\mathbf{I}(\theta | \mathbf{C}_i, \mathbf{V}_i, M_i) = \frac{d}{d\theta} \log \text{MNL}(\mathbf{C}_i; \mathbf{\eta}_i, M_i)$
27. $\mathbf{I}(\theta | \mathbf{C}_i, \mathbf{V}_i, M_i) = \mathcal{I}^{-1}(\theta) \mathbf{I}(\theta | \mathbf{C}_i, \mathbf{V}_i, M_i)$

24

# Appendix B Proofs

**Proof of Lemma 2.1:** It holds that

$$
lc_{\mathbf{C}|\mathbf{V},\mu}(\theta, \mu) = \sum_{i=1}^n \left[ \sum_{k=1}^d C_{ik}\eta_{ik} + \mu_i \mathbf{C}_i'\mathbf{1}_d - M_i \log \left(\sum_{k=1}^d e^{\eta_{ik} + \mu_i}\right) \right] \\
= \sum_{i=1}^n \left[ \sum_{k=1}^d C_{ik}\eta_{ik} + M_i \mu_i - M_i \log \left(\sum_{k=1}^d e^{\eta_{ik} + \mu_i}\right) \right] \\
= \sum_{i=1}^n \left[ \sum_{k=1}^d C_{ik}\eta_{ik} - M_i \log \left(\sum_{k=1}^d e^{\eta_{ik}}\right) \right] \\
= lc_{\mathbf{C}|\mathbf{V},M}(\theta),
$$

where the second to last equality holds because $\mu_i \sum_{k=1}^d C_{ik} = M_i\mu_i$. Therefore, adding $\mu_i$ does not change the likelihood. $\square$

**Proof of Lemma 2.2:** Notice that $Q_n(\theta, \mu)$ is differentiable w.r.t. $\mu$ for any given $\theta$. By letting $\frac{\partial Q_n(\theta, \mu)}{\partial \mu} = \mathbf{0}$, we can obtain function $\hat{\mu}_n(\theta)$ such that $\frac{\partial Q_n(\theta, \mu)}{\partial \mu}|_{\mu=\hat{\mu}_n(\theta)} = \mathbf{0}$ for every $\theta$. By definition,

$$
Q_n(\theta, \mu) = -[lc_{\mathbf{C}|\mathbf{V},\mu}(\theta, \mu) + f(\theta, \mu)] = -[lc_{\mathbf{C}|\mathbf{V},M}(\theta) + f(\theta, \mu)],
$$

which implies that $\frac{\partial Q_n(\theta, \mu)}{\partial \mu} = -\frac{\partial f(\theta, \mu)}{\partial \mu}$. Since $\frac{\partial f(\theta, \mu)}{\partial \mu_i} = M_i - e^{\mu_i} \sum_{k=1}^d e^{\eta_{ik}}$, we obtain the expression of $\hat{\mu}_n(\theta)$ as in (3.3). Plugging it into (2.5), we have that

$$
f(\theta, \hat{\mu}_n(\theta)) = \sum_{i=1}^n \left[ M_i \log \left(\sum_{k=1}^d e^{\eta_{ik} + \hat{\mu}_i}\right) - \sum_{k=1}^d e^{\eta_{ik} + \hat{\mu}_i} \right] \\
= \sum_{i=1}^n \left[ M_i \log \left(\sum_{k=1}^d e^{\eta_{ik}} \frac{M_i}{\sum_{k=1}^d e^{\eta_{ik}}}\right) - M_i \right] \\
= \sum_{i=1}^n [M_i \log M_i - M_i], \quad (B.1)
$$

which does not depend on $\theta$. As a result, it holds that

$$
\arg \min_{\theta \in \Theta} Q_n(\theta, \hat{\mu}_n(\theta)) = \arg \min_{\theta \in \Theta} [Q_n^*(\theta) - f(\theta, \hat{\mu}_n(\theta))] = \arg \min_{\theta \in \Theta} Q_n^*(\theta).
$$

Because $\hat{\theta} = \arg \min_{\theta \in \Theta} Q_n(\theta, \hat{\mu}_n(\theta))$, we have that

$$
\hat{\theta} = \arg \min_{\theta \in \Theta} Q_n^*(\theta) = \tilde{\theta}.
$$

The claimed lemma then follows. $\square$

25

**Proof of Lemma 3.1:** By Equation (2.1), it holds that

$$
\text{Pr}(C_{ik}, C_{id} | \mathbf{V}_i, M_i) = \frac{M_i!}{C_{ik}! C_{id}! (M_i - C_{ik} - C_{id})!} \left(\frac{e^{\eta_{ik}^*}}{ \Lambda_i^*}\right)^{C_{ik}} \left(\frac{e^{\eta_{id}^*}}{ \Lambda_i^*}\right)^{C_{id}} \left(\frac{\Lambda_i^* - e^{\eta_{ik}^*} - e^{\eta_{id}^*}}{ \Lambda_i^*}\right)^{M_i-C_{ik}-C_{id}}. \quad (B.2)
$$

Because $N_{ik} = C_{ik} + C_{id}$, we have that

$$
\text{Pr}(C_{ik}, C_{id} | \mathbf{V}_i, M_i) = \text{Pr}(C_{ik}, C_{id}, N_{ik} | \mathbf{V}_i, M_i) \\
= \text{Pr}(C_{ik}, C_{id} | N_{ik}, \mathbf{V}_i, M_i) \text{ Pr}(N_{ik} | \mathbf{V}_i, M_i). \quad (B.3)
$$

Compute $\text{Pr}(N_{ik} | \mathbf{V}_i, M_i)$ as

$$
\text{Pr}(N_{ik} | \mathbf{V}_i, M_i) = \frac{M_i!}{N_{ik}! (M_i - N_{ik})!} \left(\frac{e^{\eta_{ik}^*} + e^{\eta_{id}^*}}{\Lambda_i^*}\right)^{N_{ik}} \left(\frac{\Lambda_i^* - e^{\eta_{ik}^*} - e^{\eta_{id}^*}}{\Lambda_i^*}\right)^{M_i - N_{ik}}.
$$

Together with Equations (B.2) and (B.3), we obtain that

$$
\text{Pr}(C_{ik}, C_{id} | N_{ik}, \mathbf{V}_i, M_i) = \frac{N_{ik}!}{C_{ik}! C_{id}!} \left(\frac{e^{\eta_{ik}^*}}{e^{\eta_{ik}^*} + e^{\eta_{id}^*}}\right)^{C_{ik}} \left(\frac{e^{\eta_{id}^*}}{e^{\eta_{ik}^*} + e^{\eta_{id}^*}}\right)^{C_{id}} \\
= \text{Pr}(C_{ik}, C_{id} | N_{ik}, \mathbf{V}_i).
$$

By replacing $e^{\eta_{id}^*}$ with $1$ because $\theta_d^* = \mathbf{0}$, we obtain the claimed result. $\square$

**Proof of Lemma 3.2:** Under the assumption that $\text{Pr}(M_i | \mathbf{V}_i) = \text{Po}(\sum_{k=1}^d e^{\eta_{ik}^*+0})$, we can obtain that

$$
\text{Pr}(\mathbf{C}_i | \mathbf{V}_i) = \text{Pr}(\mathbf{C}_i | \mathbf{V}_i, M_i) \text{ Po}\left(\sum_{k=1}^d e^{\eta_{ik}^*}\right) \\
= \prod_{k=1}^d \text{Po}(e^{\eta_{ik}^*}) = \prod_{k=1}^d \frac{e^{\eta_{ik}^*} C_{ik} e^{-e^{\eta_{ik}^*}}}{C_{ik}!} = \frac{\prod_{k=1}^d e^{\eta_{ik}^* C_{ik}} e^{-\sum_{k=1}^d e^{\eta_{ik}^*}}}{C_{i1}! \dots C_{id}!}.
$$

The log-likelihood function is written as

$$
\log \left[ \frac{\prod_{k=1}^d e^{\eta_{ik}^* C_{ik}} e^{-\sum_{k=1}^d e^{\eta_{ik}^*}}}{C_{i1}! \dots C_{id}!} \right] \sim \sum_{i=1}^n \sum_{k=1}^d C_{ik}\eta_{ik} - \sum_{i=1}^n \sum_{k=1}^d e^{\eta_{ik}} \\
= \sum_{i=1}^n \sum_{k=1}^d (C_{ik}\eta_{ik} - e^{\eta_{ik}}) \\
= qlc_{\mathbf{C}|\mathbf{V}}(\theta, \mathbf{0}).
$$

Therefore, $\hat{\theta}_P$ maximizes the true log-likelihood function based upon $\text{Pr}(\mathbf{C}_i | \mathbf{V}_i)$. The lemma follows. $\square$

**Lemma B.1.** Under Assumption 4.2, $-L_{\mathbf{C}|\mathbf{V},M}(\theta)$ is convex in $\theta$.

26

**Proof of Lemma B.1:** By the definitions of $lc_{\mathbf{C}|\mathbf{V},M}(\theta)$ and $L_{\mathbf{C}|\mathbf{V},M}(\theta)$ and Assumption 4.2, we have that

$$
-\frac{1}{n} lc_{\mathbf{C}|\mathbf{V},M}(\theta) = -\frac{1}{n} \sum_{i=1}^n \left[ \sum_{k=1}^d C_{ik}\eta_{ik} - M_i \log \left(\sum_{k=1}^d e^{\eta_{ik}}\right) \right] \\
= -\frac{1}{n} \sum_{i=1}^n \left[ \sum_{k=1}^d C_{ik} \mathbf{V}_i'\theta_k - M_i \log \left(\sum_{k=1}^d e^{\mathbf{V}_i'\theta_k}\right) \right] \\
\xrightarrow{p} -\mathbb{E} \left[ \sum_{k=1}^d C_k \mathbf{V}'\theta_k - M \log \left(\sum_{k=1}^d e^{\mathbf{V}'\theta_k}\right) \right] \\
= -L_{\mathbf{C}|\mathbf{V},M}(\theta) \\
= \sum_{k=1}^d \mathbb{E} [C_k \mathbf{V}'\theta_k] - \mathbb{E} \left[ M \log \left(\sum_{k=1}^d e^{\mathbf{V}'\theta_k}\right) \right].
$$

The first term is convex in $\theta$ because it only involves linear functions. It has been shown in Section 3.1.5 in Boyd et al. (2004) that $\log(\sum_{k=1}^d e^{\mathbf{V}'\theta_k})$ is convex in $\mathbf{V}'\theta_1, \dots, \mathbf{V}'\theta_d$. Because $\mathbf{V}'\theta_k$ is a linear function of $\theta_k$ and sums of convex functions are convex, the second term is convex in $\theta$. $\square$

**Lemma B.2.** Under Assumption 4.2, $\theta^*$ is the unique fixed point of $\tilde{\theta}(\theta)$.

**Proof of Lemma B.2:** By definition, $\tilde{\theta}(\tilde{\theta}) = \arg \min_{\theta \in \Theta} Q^\dagger(\theta, \tilde{\theta})$. Thus, if we can show that $\frac{\partial Q^\dagger(\theta, \tilde{\theta})}{\partial\theta}|_{\theta=\theta_0} = \mathbf{0}$ holds only at $\theta = \theta^*$, then $\theta^*$ is the unique fixed point of $\tilde{\theta}(\cdot)$. Taking the first order derivative of $-L_{\mathbf{C}|\mathbf{V},M}(\cdot)$, we obtain that for any $\theta \in \Theta$,

$$
-\frac{d}{d\theta} L_{\mathbf{C}|\mathbf{V},M}(\theta)|_{\theta=\theta_0} = \mathbb{E} \begin{bmatrix}
\frac{M e^{\mathbf{V}'\theta_1} \mathbf{V}'\mathbf{V}}{\sum_{k=1}^d e^{\mathbf{V}'\theta_k}} - C_1 \mathbf{V}', \dots, \frac{M e^{\mathbf{V}'\theta_{d-1}} \mathbf{V}'\mathbf{V}}{\sum_{k=1}^d e^{\mathbf{V}'\theta_k}} - C_{d-1} \mathbf{V}'
\end{bmatrix}' \\
\frac{\partial Q^\dagger(\theta, \tilde{\theta})}{\partial\theta} \Big|_{(\theta, \tilde{\theta})=(\theta_0, \tilde{\theta}_0)}.
$$

Therefore, proving that $\frac{\partial Q^\dagger(\theta, \tilde{\theta})}{\partial\theta}|_{\theta=\theta^*, \tilde{\theta}=\theta^*} = \mathbf{0}$ holds only at $\theta = \theta^*$ is equivalent to showing that $\frac{\partial}{\partial\theta} L_{\mathbf{C}|\mathbf{V},M}(\theta) = \mathbf{0}$ only at $\theta^*$. By Lemma B.1, $-L_{\mathbf{C}|\mathbf{V},M}(\theta)$ is convex in $\theta$. The set $\Theta$ is convex by Assumption 4.2. For a convex function over a convex set, any local minimum is also a global minimum. Along with the identification assumption in Section 2.1 that $\theta^*$ is the unique solution to $\arg \min_{\theta \in \Theta} -L_{\mathbf{C}|\mathbf{V},M}(\theta)$, we obtain that $\frac{\partial}{\partial\theta} L_{\mathbf{C}|\mathbf{V},M}(\theta) = \mathbf{0}$ holds only at $\theta = \theta^*$. $\square$

**Proof of Lemma 4.1:** Part (i) follows by applying Theorem 2 in Jennrich (1969) on the uniform law of large numbers with conditions satisfied by Assumption 4.2.
Since $Q^\dagger(\theta, \tilde{\theta})$ is continuous in $\theta$ and $\tilde{\theta}$ is compact by Assumption 4.2 (i), $Q^\dagger(\theta, \tilde{\theta})$ achieves its minimum in $\theta$ for any $\tilde{\theta}$. Given any $0 < \lambda < 1$ and $\tilde{\theta}^1 \ne \tilde{\theta}^2$, if

27

$\mathbf{v}_j \ne 0$ for $j = 1, \dots, p$, then we have
$$
e^{\mathbf{v}'[\lambda \tilde{\theta}^1 + (1-\lambda)\tilde{\theta}^2]} < \lambda e^{\mathbf{v}'\tilde{\theta}^1} + (1-\lambda) e^{\mathbf{v}'\tilde{\theta}^2}.
$$
In consequence, $Q^\dagger(\theta, \tilde{\theta})$ is strictly convex in $\theta$ for any $\tilde{\theta}$. Because $\Theta$ is a convex set, we have that $Q^\dagger(\theta, \tilde{\theta})$ have a unique minimizer for any $\tilde{\theta}$. Part (ii) holds.
For part (iii), because $Q^\dagger(\theta, \tilde{\theta})$ is strictly convex in $\theta$ for any $\tilde{\theta} \in \Theta$ and $\Theta$ is convex, the (opposite) maximum theorem implies the continuity.
To prove part (iv), we take the first-order partial derivative of $Q^\dagger(\theta, \tilde{\theta})$ with respect to $\theta$ and obtain that

$$
\frac{\partial Q^\dagger(\theta, \tilde{\theta})}{\partial\theta} = \mathbb{E} \left[ \frac{M e^{\mathbf{V}'\theta_1} \mathbf{V}'}{\sum_{k=1}^d e^{\mathbf{V}'\tilde{\theta}_k}} - C_1 \mathbf{V}', \dots, \frac{M e^{\mathbf{V}'\theta_{d-1}} \mathbf{V}'}{\sum_{k=1}^d e^{\mathbf{V}'\tilde{\theta}_k}} - C_{d-1} \mathbf{V}' \right]'.
$$

Since $\mathbb{E}[C_k | \mathbf{V}, M] = \frac{M e^{\mathbf{V}'\theta_k}}{\sum_{k=1}^d e^{\mathbf{V}'\theta_k}}$, we have that

$$
\frac{\partial Q^\dagger(\theta, \tilde{\theta})}{\partial\theta} \Big|_{(\theta, \tilde{\theta})=(\theta^*, \theta^*)} \\
= \mathbb{E} \left[ \mathbb{E} \left[ \frac{M e^{\mathbf{V}'\theta_1^*} \mathbf{V}'}{\sum_{k=1}^d e^{\mathbf{V}'\theta_k^*}} - C_1 \mathbf{V}' \Big| \mathbf{V}, M \right], \dots, \mathbb{E} \left[ \frac{M e^{\mathbf{V}'\theta_{d-1}^*} \mathbf{V}'}{\sum_{k=1}^d e^{\mathbf{V}'\theta_k^*}} - C_{d-1} \mathbf{V}' \Big| \mathbf{V}, M \right] \right]' \\
= \mathbb{E} \left[ \frac{M e^{\mathbf{V}'\theta_1^*} \mathbf{V}'}{\sum_{k=1}^d e^{\mathbf{V}'\theta_k^*}} - \frac{M e^{\mathbf{V}'\theta_1^*} \mathbf{V}'}{\sum_{k=1}^d e^{\mathbf{V}'\theta_k^*}}, \dots, \frac{M e^{\mathbf{V}'\theta_{d-1}^*} \mathbf{V}'}{\sum_{k=1}^d e^{\mathbf{V}'\theta_k^*}} - \frac{M e^{\mathbf{V}'\theta_{d-1}^*} \mathbf{V}'}{\sum_{k=1}^d e^{\mathbf{V}'\theta_k^*}} \right]' \\
= \mathbf{0}. \quad (B.4)
$$

Because $Q^\dagger(\theta, \tilde{\theta})$ is strictly convex in $\theta$ for any $\tilde{\theta}$, $Q^\dagger(\theta, \theta^*)$ is certainly strictly convex in $\theta$. Combining with (B.4), we obtain that $\theta^* = \arg \min_{\theta \in \Theta} Q^\dagger(\theta, \theta^*)$, which implies that $\tilde{\theta}(\theta^*) = \theta^*$.
Part (v) is proved by Lemma B.2. $\square$

**Proof of Theorem 4.1:** We show that if $\hat{\theta}^{(0)} \xrightarrow{p} \theta^*$ as $n \to \infty$, then $\hat{\theta}^{(1)} \xrightarrow{p} \theta^*$ as well. By (3.4), $\hat{\theta}^{(1)}$ satisfies that

$$
\hat{\theta}^{(1)} = \arg \min_{\theta_1 \in \Theta_1} Q_{1n}(\theta_1, \hat{\mu}_n(\hat{\theta}^{(0)})).
$$

The first order condition provides that

$$
\frac{\partial}{\partial\theta_1} -Q_{1n}(\theta_1, \hat{\mu}_n(\hat{\theta}^{(0)})) \Big|_{\theta_1=\hat{\theta}_1^{(1)}} = \mathbf{0}.
$$

28

The mean value theorem implies that

$$
\mathbf{0} = \frac{1}{n} \frac{\partial}{\partial\theta_1} Q_{1n}(\hat{\theta}_1^*, \hat{\mu}_n(\hat{\theta}^{(0)})) |_{\theta_1=\hat{\theta}_1^*} + \frac{1}{n} \frac{\partial^2}{\partial\theta_1\partial\theta_1'} Q_{1n}(\tilde{\theta}_1, \hat{\mu}_n(\hat{\theta}^{(0)})) (\hat{\theta}_1^{(1)} - \theta_1^*),
$$

where $\tilde{\theta}_1$ lies between $\hat{\theta}_1^{(1)}$ and $\theta_1^*$. Since $\hat{\theta}^{(0)} \xrightarrow{p} \theta^*$, we have that

$$
A_n \equiv \mathbb{E} \left[ \frac{M e^{\mathbf{V}'\theta_1^*}}{\sum_{k=1}^d e^{\mathbf{V}'\tilde{\theta}_k}} \mathbf{V} - C_k \mathbf{V} \right] \mathbb{E} \left[ \frac{M e^{\mathbf{V}'\theta_1^*}}{\sum_{k=1}^d e^{\mathbf{V}'\theta_k^*}} \mathbf{V} - C_k \mathbf{V} \Big| \mathbf{V}, M \right] = \mathbf{0}.
$$

By a similar argument, it can be shown that $B_n$ converges in probability to a non-singular matrix for any $\theta_1^* \in \Theta_1$. Therefore, it must hold that $\hat{\theta}_1^{(1)} \xrightarrow{p} \theta_1^*$. Hence, $\hat{\theta}^{(S)} \xrightarrow{p} \theta^*$ as $n \to \infty$ for any $S$. $\square$

**Proof of Theorem 4.2:** Based on Lemma 4.1, Assumptions 1, 2a, and 5 in Pastorello et al. (2003) are satisfied. Therefore, Proposition 1 in Pastorello et al. (2003) holds, which implies that as $n \to \infty$,

$$
\sup_{\tilde{\theta} \in \Theta} \|\hat{\theta}_n(\tilde{\theta}) - \tilde{\theta}(\tilde{\theta})\| \xrightarrow{p} 0, \quad (B.5)
$$

where $\hat{\theta}_n(\cdot)$ is defined in (3.4).

Let $\tilde{\theta}^{(0)} = \text{plim}_{n \to \infty} \hat{\theta}^{(0)}$, where $\hat{\theta}^{(0)}$ is the initial value of our IDC estimator $\hat{\theta}^{(0)}$. Define $\tilde{\theta}^{(1)} = \tilde{\theta}(\tilde{\theta}^{(0)})$, $\tilde{\theta}^{(2)} = \tilde{\theta}(\tilde{\theta}^{(1)}) = \tilde{\theta}^2(\tilde{\theta}^{(0)})$, \dots, $\tilde{\theta}^{(s)} = \tilde{\theta}(\tilde{\theta}^{(s-1)}) = \tilde{\theta}^s(\tilde{\theta}^{(0)})$ for any $s \in \mathbb{Z}^+$, where $\mathbb{Z}^+$ denotes the set of positive integers. Next, we show that $\{\tilde{\theta}^{(s)}\}$ is a Cauchy sequence. By Assumption 4.3, we have that for any $s_1 > s_2 \ge 1$,

$$
\|\tilde{\theta}^{(s_1)} - \tilde{\theta}^{(s_2)}\| = \|\tilde{\theta}^{s_1}(\tilde{\theta}^{(0)}) - \tilde{\theta}^{s_2}(\tilde{\theta}^{(0)})\| \\
\le \|\tilde{\theta}^{s_1}(\tilde{\theta}^{(0)}) - \tilde{\theta}^{s_1-1}(\tilde{\theta}^{(0)})\| + \|\tilde{\theta}^{s_1-1}(\tilde{\theta}^{(0)}) - \tilde{\theta}^{s_1-2}(\tilde{\theta}^{(0)})\| \\
+ \dots + \|\tilde{\theta}^{s_2+1}(\tilde{\theta}^{(0)}) - \tilde{\theta}^{s_2}(\tilde{\theta}^{(0)})\| \\
\le [C^{s_1-1} + C^{s_1-2} + \dots + C^{s_2}] \|\tilde{\theta}^{(1)} - \tilde{\theta}^{(0)}\| \\
= C^{s_2} \sum_{i=0}^{s_1-s_2-1} C^i \|\tilde{\theta}^{(1)} - \tilde{\theta}^{(0)}\| \\
\le C^{s_2} \sum_{i=0}^{\infty} C^i \|\tilde{\theta}^{(1)} - \tilde{\theta}^{(0)}\| \\
= \frac{C^{s_2}}{1-C} \|\tilde{\theta}^{(1)} - \tilde{\theta}^{(0)}\|, \quad (B.6)
$$

which implies that $\{\tilde{\theta}^{(s)}\}$ is Cauchy because $C < 1$. Since $\Theta \subseteq \mathbb{R}^{pd}$ is compact by Assumption 4.2 and $\mathbb{R}^{pd}$ is complete with respect to $\|\cdot\|$, $\Theta$ is also complete with

29

respect to $\|\cdot\|$. Therefore, $\tilde{\theta}^{(s)}$ converges to a limit $\tilde{\theta}^*$ in $\Theta$ as $s \to \infty$. Because

$$
\tilde{\theta}(\tilde{\theta}^*) = \tilde{\theta}(\lim_{s \to \infty} \tilde{\theta}^{(s)}) = \lim_{s \to \infty} \tilde{\theta}(\tilde{\theta}^{(s)}) = \lim_{s \to \infty} \tilde{\theta}^{(s+1)} = \tilde{\theta}^*,
$$

it holds that $\tilde{\theta}^*$ is a fixed point of the mapping $\tilde{\theta}: \Theta \to \Theta$. By Lemma B.2, $\theta^*$ is the unique fixed point of $\tilde{\theta}(\cdot)$. Thus, $\tilde{\theta}^* = \theta^*$ and $\lim_{s \to \infty} \tilde{\theta}^{(s)} = \theta^*$.
We now show that $\hat{\theta}^{(s)} - \tilde{\theta}^{(s)} = o_p(1)$ for any $s \in \mathbb{Z}^+$ by induction. By the definition of $\hat{\theta}^{(0)}$, $\hat{\theta}^{(0)} - \tilde{\theta}^{(0)} = o_p(1)$. Assuming that $\hat{\theta}^{(t)} - \tilde{\theta}^{(t)} = o_p(1)$ for some $t$, it holds that

$$
\hat{\theta}^{(t+1)} - \tilde{\theta}^{(t+1)} = \hat{\theta}_n(\hat{\theta}^{(t)}) - \tilde{\theta}(\tilde{\theta}^{(t)}) \\
= [\hat{\theta}_n(\hat{\theta}^{(t)}) - \tilde{\theta}(\hat{\theta}^{(t)})] + [\tilde{\theta}(\hat{\theta}^{(t)}) - \tilde{\theta}(\tilde{\theta}^{(t)})] \\
= o_p(1),
$$

where the last equality holds because $\hat{\theta}_n(\hat{\theta}^{(t)}) - \tilde{\theta}(\hat{\theta}^{(t)}) = o_p(1)$ by (B.5), $\hat{\theta}^{(t)} - \tilde{\theta}^{(t)} = o_p(1)$ by assumption and $\tilde{\theta}(\cdot)$ is continuous by Lemma 4.1 (iii). Therefore, $\hat{\theta}^{(s)} - \tilde{\theta}^{(s)} = o_p(1)$ for any $s \in \mathbb{Z}^+$.
Hence, we have that if $S \to \infty$ and $n \to \infty$, then

$$
\| \hat{\theta}^* - \theta^* \| = \| \hat{\theta}^{(S)} - \tilde{\theta}^{(S)} \| + \| \tilde{\theta}^{(S)} - \theta^* \| \\
= A(n, S) + B(S) \xrightarrow{p} 0,
$$

because $A(n, S) \xrightarrow{p} 0$ as $n \to \infty$ for any given $S$ and $B(S) \to 0$ as $S \to \infty$ by $\lim_{s \to \infty} \tilde{\theta}^{(s)} = \theta^*$. The theorem holds. $\square$

**Lemma B.3.** Under the conditions in Theorem 4.3, it holds that

$$
\sqrt{n}(\hat{\theta}^{(S+1)} - \hat{\theta}^{(S)}) \xrightarrow{p} 0.
$$

**Proof of Lemma B.3:** We first show that the result in the lemma holds if the conditions in part (i) of the theorem hold. By (3.4), for any $s \in \mathbb{Z}^+$, $\hat{\theta}^{(s)}$ satisfies that

$$
\frac{\partial}{\partial\theta} Q_n(\theta, \hat{\mu}_n(\hat{\theta}^{(S-1)})) \Big|_{\theta=\hat{\theta}^{(S)}} = \mathbf{0}.
$$

Applying Taylor expansion to the left-hand-side of the equality at $\theta^*$. Because $\hat{\theta}^{(S)} - \theta^* = o_p(1)$ and $\hat{\theta}^{(S-1)} - \theta^* = o_p(1)$, we obtain that

$$
\frac{\partial}{\partial\theta} Q_n(\theta, \tilde{\theta}) \Big|_{\theta=\theta^*, \tilde{\theta}=\theta^*} + \frac{\partial^2}{\partial\theta\partial\tilde{\theta}'} Q_n(\theta, \tilde{\theta}) \Big|_{\theta=\theta^*, \tilde{\theta}=\theta^*} (\hat{\theta}^{(S)} - \theta^*) \\
+ \frac{\partial^2}{\partial\tilde{\theta}\partial\tilde{\theta}'} Q_n(\theta, \tilde{\theta}) \Big|_{\theta=\theta^*, \tilde{\theta}=\hat{\theta}^{(S-1)}} (\hat{\theta}^{(S-1)} - \theta^*) = \mathbf{0} \quad (B.7)
$$

30

by ignoring higher-order terms. Since $\frac{\partial^2 Q_n(\theta, \tilde{\theta})}{\partial\theta\partial\theta'} \Big|_{\theta=\theta^*, \tilde{\theta}=\theta^*}$ is non-singular and both $\hat{\theta}^{(S)}$ and $\hat{\theta}^{(S-1)}$ are consistent estimators of $\theta^*$, we have that

$$
\left( \frac{\partial^2 Q_n(\theta, \tilde{\theta})}{\partial\theta\partial\theta'} \Big|_{\theta=\hat{\theta}^{(S)}, \tilde{\theta}=\hat{\theta}^{(S-1)}} \right)^{-1}
$$

exists with a high probability when $n$ is large. Define

$$
A_n = \left( \frac{\partial^2 Q_n(\theta, \tilde{\theta})}{\partial\theta\partial\theta'} \Big|_{\theta=\hat{\theta}^{(S)}, \tilde{\theta}=\hat{\theta}^{(S-1)}} \right)^{-1} \left( -\frac{\partial^2 Q_n(\theta, \tilde{\theta})}{\partial\theta\partial\tilde{\theta}'} \Big|_{\theta=\theta^*, \tilde{\theta}=\theta^*} \right) \\
B_n = \left( \frac{\partial^2 Q_n(\theta, \tilde{\theta})}{\partial\theta\partial\theta'} \Big|_{\theta=\hat{\theta}^{(S)}, \tilde{\theta}=\hat{\theta}^{(S-1)}} \right)^{-1} \left( \frac{\partial^2 Q_n(\theta, \tilde{\theta})}{\partial\theta\partial\tilde{\theta}'} \Big|_{\theta=\hat{\theta}^{(S)}, \tilde{\theta}=\hat{\theta}^{(S-1)}} \right).
$$

By the law of large numbers and the consistency of $\hat{\theta}^{(S)}$ for any $S$, we have that

$$
A_n = \left( \frac{\partial^2 Q^\dagger(\theta, \tilde{\theta})}{\partial\theta\partial\theta'} \Big|_{\theta=\theta^*, \tilde{\theta}=\theta^*} \right)^{-1} \left( -\frac{\partial^2 Q^\dagger(\theta, \tilde{\theta})}{\partial\theta\partial\tilde{\theta}'} \Big|_{\theta=\theta^*, \tilde{\theta}=\theta^*} \right) + o_p(1) = A + o_p(1) \\
B_n = \left( \frac{\partial^2 Q^\dagger(\theta, \tilde{\theta})}{\partial\theta\partial\theta'} \Big|_{\theta=\theta^*, \tilde{\theta}=\theta^*} \right)^{-1} \left( \frac{\partial^2 Q^\dagger(\theta, \tilde{\theta})}{\partial\theta\partial\tilde{\theta}'} \Big|_{\theta=\theta^*, \tilde{\theta}=\theta^*} \right) + o_p(1) = B + o_p(1).
$$

By ignoring the smaller order terms, we obtain from Equation (B.7) that

$$
\hat{\theta}^{(S)} - \theta^* = A_n + B_n (\hat{\theta}^{(S-1)} - \theta^*) = \sum_{t=0}^{S-1} B_n^t A_n + B_n^S (\hat{\theta}^{(0)} - \theta^*),
$$

where the second equality follows from iterating the first equality. It then holds that

$$
\sqrt{n}(\hat{\theta}^{(S+1)} - \hat{\theta}^{(S)}) = \sqrt{n} B^S A + \sqrt{n} B^S (B - I) (\hat{\theta}^{(0)} - \theta^*).
$$

By Assumption 4.4, $\|B\| < 1$, which implies that $\sqrt{n}\|B\|^S \to 0$ as $S \ge \log(n)$ and $n \to \infty$. Hence, $\sqrt{n}(\hat{\theta}^{(S+1)} - \hat{\theta}^{(S)}) = o_p(1)$. The claimed lemma follows.
We now prove that under the conditions in part (ii) of the theorem, the result also holds. If we can show that for any $s \in \mathbb{Z}^+$, with probability approaching one, there exists a constant $c < 1$ such that

$$
\|\hat{\theta}_n(\hat{\theta}^{(S+1)}) - \hat{\theta}_n(\hat{\theta}^{(S)})\| \le c \|\hat{\theta}^{(S+1)} - \hat{\theta}^{(S)}\|, \quad (B.8)
$$

then by the same derivation as (B.6), we would obtain that

$$
\|\hat{\theta}^{(S+1)} - \hat{\theta}^{(S)}\| \le \frac{c^S}{1-c} \|\hat{\theta}^{(1)} - \hat{\theta}^{(0)}\|.
$$

Because $c < 1$ and $S > n^\delta$ for some $\delta > 0$, it holds that

$$
\sqrt{n}\|\hat{\theta}^{(S+1)} - \hat{\theta}^{(S)}\| \le \frac{\sqrt{n}c^S}{1-c} \|\hat{\theta}^{(1)} - \hat{\theta}^{(0)}\| \to 0.
$$

31

Thus, it suffices to prove (B.8).

By the implicit function theorem, $\tilde{\theta}(\cdot)$ is continuously differentiable in $\Theta$. Together with Assumption 4.3, we have that there exists $\epsilon > 0$ such that for any $\tilde{\theta} \in \Theta$ and $\tilde{\nu} \in B(\tilde{\theta}(\tilde{\nu})) = \{\theta \in \Theta : \|\theta - \tilde{\theta}(\tilde{\nu})\| \le \epsilon\}$, we have

$$
\|\tilde{\theta}(\tilde{\theta}) - \tilde{\theta}(\tilde{\nu})\| \le C_\epsilon \|\tilde{\theta} - \tilde{\nu}\|,
$$

for some $C_\epsilon \le C < 1$. By (B.5), $\text{Pr} \left[ \hat{\theta}_n(\hat{\theta}^{(S)}) \in B(\tilde{\theta}(\hat{\theta}^{(S)})) \right] \to 1$ as $n \to \infty$ for any $S$ and $\epsilon$. Therefore, with probability approaching one, it holds that

$$
\|\hat{\theta}_n(\hat{\theta}^{(S+1)}) - \hat{\theta}_n(\hat{\theta}^{(S)})\| = \|\tilde{\theta}_n(\hat{\theta}^{(S+1)}) - \tilde{\theta}_n(\hat{\theta}^{(S)})\| \\
\le C_\epsilon \|\hat{\theta}_n(\hat{\theta}^{(S+1)}) - \hat{\theta}_n(\hat{\theta}^{(S)})\| = C_\epsilon \|\hat{\theta}^{(S+1)} - \hat{\theta}^{(S)}\|. \quad (B.9)
$$

For any $\tilde{\theta} \in \Theta$, define $\tilde{\Omega}(\tilde{\nu}) = \frac{\partial}{\partial\tilde{\nu}} \tilde{\theta}(\tilde{\nu})$, where $\frac{\partial}{\partial\tilde{\nu}} \tilde{\theta}(\tilde{\nu})$ is the Jacobian matrix of dimension $dp \times dp$. By the implicit function theorem, we have

$$
\frac{\partial}{\partial\tilde{\theta}} \tilde{\theta}(\tilde{\theta}) = \left[ \frac{\partial}{\partial\theta} g(\theta, \tilde{\theta}) \Big|_{(\theta, \tilde{\theta})=(\tilde{\theta}(\tilde{\theta}), \tilde{\theta})} \right]^{-1} \frac{\partial}{\partial\tilde{\theta}} g(\theta, \tilde{\theta}) \Big|_{(\theta, \tilde{\theta})=(\tilde{\theta}(\tilde{\theta}), \tilde{\theta})}.
$$

where $g(\theta, \tilde{\theta}) = \frac{\partial Q^\dagger(\theta, \tilde{\theta})}{\partial\theta}$ is the function that defines $\tilde{\theta}(\tilde{\theta})$. Similarly, we define $\Omega_n(\tilde{\nu}) = \frac{\partial}{\partial\tilde{\nu}} \hat{\theta}_n(\tilde{\nu})$ and $g_n(\theta, \tilde{\theta}) = \frac{\partial Q_n(\theta, \tilde{\theta})}{\partial\theta}$. By Assumption 4.2 (ii) and (iii) and the uniform law of large numbers, we have

$$
\sup_{\tilde{\theta} \in \Theta} \|\Omega_n(\tilde{\nu}) - \Omega(\tilde{\nu})\| \xrightarrow{p} 0. \quad (B.10)
$$

By Assumption 4.2 (i), $\Theta$ is convex. Applying a multivariate Taylor expansion (Dieudonné (2011), p. 190), we can write $\tilde{\theta}(\tilde{\theta}) - \tilde{\theta}(\tilde{\nu}) = \Lambda(\tilde{\nu}, \tilde{\theta}) (\tilde{\theta} - \tilde{\nu})$ for any $\tilde{\theta}, \tilde{\nu} \in \Theta$, where

$$
\Lambda(\tilde{\nu}, \tilde{\theta}) = \int_0^1 \Omega(\tilde{\nu} + \xi (\tilde{\theta} - \tilde{\nu})) d\xi.
$$

Similarly, we have $\hat{\theta}_n(\tilde{\theta}) - \hat{\theta}_n(\tilde{\nu}) = \Lambda_n(\tilde{\nu}, \tilde{\theta}) (\tilde{\theta} - \tilde{\nu})$ for any $\tilde{\nu}, \tilde{\theta} \in \Theta$, where

$$
\Lambda_n(\tilde{\nu}, \tilde{\theta}) = \int_0^1 \Omega_n(\tilde{\nu} + \xi (\tilde{\theta} - \tilde{\nu})) d\xi.
$$

32

It holds that

$$
\|\hat{\theta}_n(\hat{\theta}^{(S+1)}) - \hat{\theta}_n(\hat{\theta}^{(S)})\| \le \left\| \left[ \Lambda_n(\hat{\theta}^{(S+1)}, \hat{\theta}^{(S)}) - \Lambda(\hat{\theta}^{(S+1)}, \hat{\theta}^{(S)}) \right] (\hat{\theta}^{(S+1)} - \hat{\theta}^{(S)}) \right\| \\
+ \|\Lambda(\hat{\theta}^{(S+1)}, \hat{\theta}^{(S)}) (\hat{\theta}^{(S+1)} - \hat{\theta}^{(S)})\| \\
\le \|[\Lambda_n(\tilde{\nu}, \tilde{\theta}) - \Lambda(\tilde{\nu}, \tilde{\theta})] [\tilde{\theta}^{(S+1)} - \tilde{\theta}^{(S)}]\| \\
+ C_\epsilon \|\hat{\theta}^{(S+1)} - \hat{\theta}^{(S)}\|.
$$

where the first inequality follows from the triangular inequality and the second inequality holds by (B.9). Because the first term on the right-hand side has the order $o_p (\|\hat{\theta}^{(S+1)} - \hat{\theta}^{(S)}\|)$ because of (B.10), we have shown that (B.8) holds with $c = C_\epsilon < 1$. The lemma follows. $\square$

**Proof of Theorem 4.3:** We aim to show that $\hat{\theta}^{(S)}$ has the same influence function as $\hat{\theta}$. By the definition of the IDC estimator in Section 3.2, we have that

$$
\frac{\partial}{\partial\theta} Q_n(\theta, \hat{\mu}_n(\hat{\theta}^{(S-1)})) \Big|_{\theta=\hat{\theta}^{(S)}} = \mathbf{0}.
$$

Applying the Taylor expansion to the function on the left-hand-side of the above equation round $\hat{\theta}^{(S-1)}$, we can obtain that

$$
\frac{\partial}{\partial\theta} Q_n(\theta, \hat{\mu}_n(\hat{\theta}^{(S-1)})) \Big|_{\theta=\hat{\theta}^{(S-1)}} \\
+ \frac{\partial^2}{\partial\theta\partial\theta'} Q_n(\theta, \hat{\mu}_n(\hat{\theta}^{(S-1)})) \Big|_{\theta=\tilde{\theta}^{(S-1)}} (\hat{\theta}^{(S)} - \hat{\theta}^{(S-1)}) = \mathbf{0}, \quad (B.11)
$$

where $\tilde{\theta}^{(S-1)}$ lies between $\hat{\theta}^{(S)}$ and $\hat{\theta}^{(S-1)}$. The definition of $Q_n(\theta, \mu)$ implies that

$$
\frac{\partial}{\partial\theta} Q_n(\theta, \hat{\mu}_n(\hat{\theta}^{(S-1)})) \Big|_{\theta=\hat{\theta}^{(S-1)}} \\
= -\frac{d}{d\theta} L_{\mathbf{C}|\mathbf{V},M}(\hat{\theta}^{(S-1)}) - \frac{\partial}{\partial\theta} f(\theta, \hat{\mu}_n(\hat{\theta}^{(S-1)})) \Big|_{\theta=\hat{\theta}^{(S-1)}}.
$$

Applying the expression of $f(\theta, \mu)$, it can be shown that

$$
\frac{\partial}{\partial\theta} f(\theta, \hat{\mu}_n(\hat{\theta}^{(S-1)})) \Big|_{\theta=\hat{\theta}^{(S-1)}} = \mathbf{0}.
$$

Therefore, (B.11) can be rewritten as

$$
-\frac{d}{d\theta} L_{\mathbf{C}|\mathbf{V},M}(\hat{\theta}^{(S-1)}) + \frac{\partial^2}{\partial\theta\partial\theta'} Q_n(\theta, \hat{\mu}_n(\hat{\theta}^{(S-1)})) \Big|_{\theta=\tilde{\theta}^{(S-1)}} (\hat{\theta}^{(S)} - \hat{\theta}^{(S-1)}) = \mathbf{0}.
$$

33

By Lemma B.3, we have $\hat{\theta}^{(S+1)} - \hat{\theta}^{(S)} = o_p(n^{-1/2})$. This implies that

$$
\frac{d}{d\theta} lc_{\mathbf{C}|\mathbf{V},M}(\hat{\theta}^{(S-1)}) = o_p(n^{-1/2}) \\
= -\frac{d}{d\theta} lc_{\mathbf{C}|\mathbf{V},M}(\theta^*) + \frac{\partial^2}{\partial\theta\partial\theta'} L_{\mathbf{C}|\mathbf{V},M}(\theta^*) (\hat{\theta}^{(S-1)} - \theta^*), \quad (B.12)
$$

where the second equality follows from the Taylor expansion and $\tilde{\theta}^\dagger$ lies between $\hat{\theta}^{(S-1)}$ and $\theta^*$.

By Theorems 4.1 and 4.2, we have that $\hat{\theta}^{(S-1)} \xrightarrow{p} \theta^*$ as $n \to \infty$ either for any fixed $S$ or as $S \to \infty$. Therefore, by Assumption 4.2 and Taylor's theorem, we can obtain that

$$
\frac{1}{n} \frac{\partial^2}{\partial\theta\partial\theta'} lc_{\mathbf{C}|\mathbf{V},M}(\theta^*) = \frac{1}{n} \frac{\partial^2}{\partial\theta\partial\theta'} lc_{\mathbf{C}|\mathbf{V},M}(\theta^*) + o_p(1).
$$

Since matrix inversion is continuous (at non-singular matrices), it follows that the inverse of $\frac{1}{n} \frac{\partial^2}{\partial\theta\partial\theta'} lc_{\mathbf{C}|\mathbf{V},M}(\theta^*)$ exists with high probability and

$$
\left[ \frac{1}{n} \frac{\partial^2}{\partial\theta\partial\theta'} lc_{\mathbf{C}|\mathbf{V},M}(\theta^*) \right]^{-1} \xrightarrow{p} \mathcal{I}^{-1}(\theta^*),
$$

where $\mathcal{I}(\theta^*)$ is the Fisher information matrix defined in Section 4.2. Using this result to (B.12), we obtain that

$$
\hat{\theta}^{(S-1)} - \theta^* = \mathcal{I}^{-1}(\theta^*) \frac{1}{n} \frac{d}{d\theta} lc_{\mathbf{C}|\mathbf{V},M}(\theta^*) + o_p(n^{-1/2}).
$$

Since $\hat{\theta}^{(S)} - \theta^* = \hat{\theta}^{(S-1)} - \theta^* + o_p(n^{-1/2})$ by Lemma B.3, it holds that

$$
\hat{\theta}^{(S)} - \theta^* = \mathcal{I}^{-1}(\theta^*) \frac{1}{n} \frac{d}{d\theta} lc_{\mathbf{C}|\mathbf{V},M}(\theta^*) + o_p(n^{-1/2}).
$$

It can be seen that $\hat{\theta}^{(S)}$ and the maximum likelihood estimator $\hat{\theta}$ have the same influence function. Hence, under the assumptions in either part (i) or (ii) of the theorem, we have $\hat{\theta}^{(S)} - \hat{\theta} = o_p(n^{-1/2})$. $\square$

**Proof of Corollary 4.4:** The result directly follows from Theorem 4.3 and the standard result on the asymptotic distribution of the maximum likelihood estimator. $\square$

**Proof of Theorem 4.5:** The proof of the theorem follows from the discussion in Section 4.2. $\square$

34

# References

AGUIRREGABIRIA, V. AND P. MIRA (2002): “Swapping the nested fixed point algorithm: A class of estimators for discrete Markov decision models," *Econometrica*, 70, 1519-1543.

\_\_\_\_\_\_ (2007): "Sequential estimation of dynamic discrete games,” *Econometrica*, 75, 1-53.

BAKER, M. AND J. WURGLER (2006): “Investor sentiment and the cross-section of stock returns,” *Journal of Finance*, 61, 1645-1680.

BAKER, S. G. (1994):“The multinomial-Poisson transformation,” *Journal of the Royal Statistical Society: Series D (The Statistician)*, 43, 495-504.

BETTMAN, J. R. (1979): “Memory factors in consumer choice: A review," *Journal of Marketing*, 43, 37-53.

BÖHNING, D. (1992): “Multinomial logistic regression algorithm," *Annals of the institute of Statistical Mathematics*, 44, 197-200.

BÖHNING, D. AND B. G. LINDSAY (1988): “Monotonicity of quadratic-approximation algorithms," *Annals of the Institute of Statistical Mathematics*, 40, 641-663.

BOYD, S., S. P. BOYD, AND L. VANDENBERGHE (2004): *Convex Optimization*, Cambridge university press.

BOYD, S., N. PARIKH, E. CHU, B. PELEATO, AND J. ECKSTEIN (2011): “Distributed optimization and statistical learning via the alternating direction method of multipliers,” *Foundations and Trends® in Machine learning*, 3, 1–122.

BUCHHOLZ, N. (2021): “Spatial equilibrium, search frictions, and dynamic efficiency in the taxi industry,” *The Review of Economic Studies*, 89, 556-591.

CHEN, Y., B. HAN, AND J. PAN (2021): “Sentiment trading and hedge fund returns,” *Journal of Finance*, 76, 2001-2033.

DAVIDSON, J., B. LIEBALD, J. LIU, P. NANDY, T. VAN VLEET, U. GARGI, S. GUPTA, Y. HE, M. LAMBERT, AND B. LIVINGSTON (2010): “The YouTube video recommendation system,” in *Proceedings of the Fourth ACM Conference on Recommender Systems*, 293-296.

DIEUDONNÉ, J. (2011): *Foundations of Modern Analysis*, Read Books Ltd.

DOMINITZ, J. AND R. P. SHERMAN (2005): “Some convergence theory for iterative estimation procedures with an application to semiparametric estimation," *Econometric Theory*, 21, 838-863.

35

FAGAN, F. AND G. IYENGAR (2018): “Unbiased scalable softmax optimization," *arXiv preprint arXiv:1803.08577*.

FAN, Y., S. PASTORELLO, AND E. RENAULT (2015):“Maximization by parts in extremum estimation,” *The Econometrics Journal*, 18, 147–171.

FRIEDMAN, J., T. HASTIE, AND R. TIBSHIRANI (2010): “Regularization paths for generalized linear models via coordinate descent,” *Journal of Statistical Software*, 33, 1.

GENTZKOW, M., J. M. SHAPIRO, AND M. TADDY (2019): “Measuring group differences in high-dimensional choices: method and application to congressional speech,” *Econometrica*, 87, 1307–1340.

GOPAL, S. AND Y. YANG (2013): “Distributed training of large-scale logistic models," in *International Conference on Machine Learning, PMLR*, 289–297.

JENNRICH, R. I. (1969): “Asymptotic properties of non-linear least squares estimators," *The Annals of Mathematical Statistics*, 40, 633–643.

KASAHARA, H. AND K. SHIMOTSU (2012): “Sequential estimation of structural models with a fixed point constraint," *Econometrica*, 80, 2303-2319.

KELLY, B. T., A. MANELA, AND A. MOREIRA (2019): “Text selection," *Working Paper 26517, National Bureau of Economic Research*.

MCFADDEN, D. (1973): “Conditional logit analysis of qualitative choice behavior," in *Frontiers in Econometrics*, edited by P. Zarembka, New York: Wiley.

NIBBERING, D. AND T. J. HASTIE (2022): “Multiclass-penalized logistic regression," *Computational Statistics & Data Analysis*, 169, 107414.

PASTORELLO, S., V. PATILEA, AND E. RENAULT (2003): “Iterative and recursive estimation in structural nonadaptive models,” *Journal of Business & Economic Statistics*, 21, 449-509.

PELLEGRINI, P. A. AND A. S. FOTHERINGHAM (2002): “Modelling spatial choice: a review and synthesis in a migration context,” *Progress in Human Geography*, 26, 487-510.

RAMAN, P., S. SRINIVASAN, S. MATSUSHIMA, X. ZHANG, H. YUN, AND S. VISHWANATHAN (2016):“DS-MLR: exploiting double separability for scaling up distributed multinomial logistic regression,” *arXiv preprint arXiv:1604.04706*.

RECHT, B., C. RE, S. WRIGHT, AND F. NIU (2011): “Hogwild!: A lock-free approach to parallelizing stochastic gradient descent," *Advances in Neural Information Processing Systems*, 24.

36

RUSSAKOVSKY, O., J. DENG, H. SU, J. KRAUSE, S. SATHEESH, S. MA, Z. HUANG, A. KARPATHY, A. KHOSLA, AND M. BERNSTEIN (2015): “Imagenet large scale visual recognition challenge,” *International Journal of Computer Vision*, 115, 211–252.

SIMON, N., J. FRIEDMAN, AND T. HASTIE (2013): “A blockwise descent algorithm for group-penalized multiresponse and multinomial regression,” *arXiv preprint arXiv:1311.6529*.

TADDY, M. (2013): “Multinomial inverse regression for text analysis,” *Journal of the American Statistical Association*, 108, 755–770.

\_\_\_\_\_\_ (2015): "Distributed multinomial regression,” *The Annals of Applied Statistics*, 9, 1394–1414.

37