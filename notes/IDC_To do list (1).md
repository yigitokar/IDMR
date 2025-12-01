# To-Do List
October 24, 2025

Based on the referee report, we need to conduct more simulation study and include an empirical analysis. Here are all the related comments.

1.  While the Monte Carlo simulations validate the finite-sample performance of the IDC estimator, an empirical application involving large choice sets (beyond `d = 150` in the paper) would strengthen the paper's practical relevance. For example, is your IDC estimator feasible on the publicly-available Yelp dataset used in Taddy (2015), where `n = 215,879`, `d = 13,938`, and `p = 11,940`? If so, does it arrive at different insights? Although the authors mentioned the issue of high-dimensional co-variates is being worked on in a companion paper, perhaps they can at least demonstrate their approach using a simplified version of Taddy's Yelp example with fewer covariates. Implementing both the IDC and Taddy's DMR on a common dataset would hopefully showcase the computational and inferential advantages of your approach.

2.  While the paper compares IDC to MLE, it would benefit from an in-depth discussion of how it compares to other modern large-scale multinomial regression methods that are currently deployed in practice, such as stochastic gradient-based methods. In a footnote, the authors mentioned that, unlike stochastic gradient descent, the IDC estimator does not inherently involve any tuning parameter. I feel a more thorough comparison is warranted. Besides, the number of iterations, S, in your algorithm can be viewed as a tuning parameter. Modern SGD variants can also leverage distributed computing frameworks in addition to GPU acceleration (e.g., PyTorch, TensorFlow), potentially offering competitive performance – particularly when the number of individuals n, is very large. Overall, I think it would strengthen the paper if the authors compare IDC with other existing approaches within the Monte Carlo simulation, as well as appealing to theoretical arguments of IDC's provable properties versus SGD's limitations (e.g., sensitivity to learning rates, lack of efficiency guarantees).

3.  In my view, a primary concern is the robustness of the MLE in large MNL models. When dealing with a large number of parameters, an efficient estimator may not perform well with a finite sample size, as discussed in James & Stein (1961).

It might be beneficial to introduce additional bias to reduce variance, such as using shrinkage estimation. While focusing on the computational challenges of traditional MLE is acceptable, it is also important to point out issues related to the large dimensionality of the parameter space. I am not suggesting to reconsider a robust estimator in this context, but I think the authors might want to discuss it somewhere in the paper. By the way, should cases like `d x n` or `d ≫ n` be explicitly ruled out? (By the first comment of the co-editor, we do not need to provide any theoretical result.)

# 1 Simulation

We need additional simulation designs.

1.  Increase `d`. We may need to let `d = 250, 500, 1000, 2000, 5000`. We can keep `p` unchanged in this design. We do not need to include MLE for these large values of `d` (maybe still do MLE when `d = 250`). We do not need to repeat five hundred times. I suggest that **we try ten repetitions first**. If the result is good and we have time, we can try fifty. **We fix n to be 1000**. The pattern for different `n` has been studied in the paper. **For both DGPs, we may increase M to accommodate large d**. For example, **we can let M to draw from a discrete uniform distribution [200, 300]**. Here are the tables that we can present after this simulation exercise. One referee asks about the optimal choice of the steps, `S`. I need to think about how to reply to that and explain this well in the paper. But since we naturally have `θ^(s)` for each step, we can first store all the values (MSE and time) for `S = 11, 12, 13, . . ., 20` and then decide what to report. We can also use stopping criteria such as `||θ^(s+1) – θ^(s)|| < ε`.

| DGP | S  | d = 250 (MSE,Time) | d = 500 (MSE,Time) | d = 1000 (MSE,Time) | d = 2000 (MSE,Time) | d = 5000 (MSE,Time) |
| :-- | :-- | :------------------- | :------------------- | :-------------------- | :-------------------- | :-------------------- |
| A   | 10 | (XX,XX)            | (XX,XX)            | (XX,XX)             | (XX,XX)             | (XX,XX)             |
|     | 20 |                    |                    |                     |                     |                     |
| C   | 10 |                    |                    |                     |                     |                     |
|     | 20 |                    |                    |                     |                     |                     |

Table I: Finite sample performance of IDC estimator with large `d`

2.  Include other stochastic gradient descent methods. We can apply the SGD to DGP-A and obtain the a table like the following. Set `n = 1000`. Let `M1` and `M2` be two of the many possible stochastic gradient descent methods. We can add more if there are more commonly used ones. SGD usually requires tuning parameters (denoted as `κ`). We want to show that the result of SGD can be sensitive to the tuning parameters. That is why for each SGD, we may try different values of tuning parameters. We shall then do the same thing for DGP-C, where some probabilities are close or very close to zero.

| DGP | SGD | κ | d = 250 (MSE,Time) | d = 500 (MSE,Time) | d = 1000 (MSE,Time) | d = 2000 (MSE,Time) | d = 5000 (MSE,Time) |
| :-- | :-- | :- | :------------------- | :------------------- | :-------------------- | :-------------------- | :-------------------- |
| A   | M1  | X | (XX,XX)            | (XX,XX)            | (XX,XX)             | (XX,XX)             | (XX,XX)             |
|     |     | X |                    |                    |                     |                     |                     |
|     |     | X |                    |                    |                     |                     |                     |
|     | M2  | X |                    |                    |                     |                     |                     |
|     |     | X |                    |                    |                     |                     |                     |
|     |     | X |                    |                    |                     |                     |                     |
| C   | M1  | X |                    |                    |                     |                     |                     |
|     |     | X |                    |                    |                     |                     |                     |
|     |     | X |                    |                    |                     |                     |                     |
|     | M2  | X |                    |                    |                     |                     |                     |
|     |     | X |                    |                    |                     |                     |                     |
|     |     | X |                    |                    |                     |                     |                     |

Table II: Finite sample performance of SGD with large `d`

3.  Adding a regularization term. Let `λ ∈ ℝ` be the tuning parameter. The general form of the regularization term is expressed as

    $L (\lambda) = \sum_{k=1}^{d-1}(\lambda \theta_{k,1} + ... + \lambda|\theta_{k,p}|)$

    We keep `λ` the same across iterations and keep the current iteration procedure. That is, we add `L (λ)` to the objective function in the initial estimation and all the iteration steps. Everything else stays the same. In the end, we want a table like this.

| n = 1000 | p = 50 (MSE,Time) | p = 100 (MSE,Time) | p = 500 (MSE,Time) | p = 1000 (MSE,Time) | p = 2000 (MSE,Time) |
| :------- | :------------------ | :------------------- | :------------------- | :-------------------- | :-------------------- |
| d        |                     |                      |                      |                       |                       |
| 200      | (XX,XX)           | (XX,XX)            | (XX,XX)            | (XX,XX)             | (XX,XX)             |
| 250      |                     |                      |                      |                       |                       |
| 500      |                     |                      |                      |                       |                       |
| 1000     |                     |                      |                      |                       |                       |
| 2000     |                     |                      |                      |                       |                       |

Table III: Finite sample performance of IDC estimator as `p` and `d` change

## 1.1 Computational Resource

To make the simulation faster, we may need extra computational power. I can use my research fund to pay for more CPU instances.

# 2 Empirical Analysis

Once the simulation, especially Part 3 where a regularization term is added, is done, we can use the data in Taddy to conduct the empirical analysis. Depending on the result of Part 3, we can use the original setup or reduce `d` and `p`.