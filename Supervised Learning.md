- [Recall: Linear Regression](#recall--linear-regression)
    + [Hypothesis:](#hypothesis-)
    + [Cost Function](#cost-function)
- [Multidimensional Inputs](#multidimensional-inputs)
    + [Notation](#notation)
- [Multivariate Linear Regression](#multivariate-linear-regression)
    + [Hypothesis](#hypothesis)
    + [Cost Function](#cost-function-1)
    + [Goal](#goal)
  * [How? Two potential solutions](#how--two-potential-solutions)
    + [Gradient descent (or other iterative algorithm)](#gradient-descent--or-other-iterative-algorithm-)
    + [Direct minimization](#direct-minimization)
- [Indirect Solution for Linear Regression](#indirect-solution-for-linear-regression)
  * [Gradient Descent Algorithm](#gradient-descent-algorithm)
  * [Gradient Descent: Intuition](#gradient-descent--intuition)
  * [2-dimensional parameters](#2-dimensional-parameters)
  * [Gradient Descent for Least Squares Cost](#gradient-descent-for-least-squares-cost)
- [Feature Normalization](#feature-normalization)
- [Direct Solution for Linear Regression](#direct-solution-for-linear-regression)
    + [Want to minimize SSD](#want-to-minimize-ssd)
    + [Find minima of function](#find-minima-of-function)
- [Direct solution](#direct-solution)
    + [Re-write SSD using vector matrix notation](#re-write-ssd-using-vector-matrix-notation)
    + [where](#where)
    + [Solution: Normal Equation](#solution--normal-equation)
- [Derivation of Normal Equations](#derivation-of-normal-equations)
    + [SSE in matrix form](#sse-in-matrix-form)
    + [Take gradient with respect to $\theta$ (vector), set to 0](#take-gradient-with-respect-to---theta---vector---set-to-0)
- [Trade-offs](#trade-offs)
  * [Gradient Descent](#gradient-descent)
  * [Normal Equations](#normal-equations)
- [Maximum Likelihood Principle (ML)](#maximum-likelihood-principle--ml-)
  * [So far, we have treated outputs as noiseless](#so-far--we-have-treated-outputs-as-noiseless)
  * [How to model uncertainty in data?](#how-to-model-uncertainty-in-data-)
    + [Hypothesis](#hypothesis-1)
    + [New cost function](#new-cost-function)
  * [Recall: Cost Function](#recall--cost-function)
  * [Alternative view: Maximum Likelihood](#alternative-view--maximum-likelihood)
  * [Maximum Likelihood: Coin Toss Example](#maximum-likelihood--coin-toss-example)
  * [Maximum Likelihood: Normal Distribution Example](#maximum-likelihood--normal-distribution-example)
  * [Maximum likelihood way of estimating model parameters $\theta$](#maximum-likelihood-way-of-estimating-model-parameters---theta-)
    + [i.i.d. Observations](#iid-observations)
- [Maximum Likelihood for Linear Regression](#maximum-likelihood-for-linear-regression)
  * [Recall: linear regression](#recall--linear-regression)
    + [Probability of of one data point {$x,t$}](#probability-of-of-one-data-point---x-t--)
    + [Max. likelihood solution](#max-likelihood-solution)
    + [Want to maximize](#want-to-maximize)
    + [Easier to maximize log()](#easier-to-maximize-log--)
    + [Want to *maximize* w.r.t. $\theta$](#want-to--maximize--wrt---theta-)
    + [But this is same as *minimizing* sum-of-squares cost](#but-this-is-same-as--minimizing--sum-of-squares-cost)
    + [Which is the same as our SSE cost from before!!](#which-is-the-same-as-our-sse-cost-from-before--)
- [Probabilistic Motivation for SSE](#probabilistic-motivation-for-sse)
- [Why is $\beta$ useful for?](#why-is---beta--useful-for-)
- [Predictive Distribution](#predictive-distribution)
- [Non-linear Features](#non-linear-features)
    + [Example](#example)
  * [Non-linear Basis Functions](#non-linear-basis-functions)
- [Polynomial basis functions](#polynomial-basis-functions)
- [Classification](#classification)
  * [What to do if data is nonlinear?](#what-to-do-if-data-is-nonlinear-)
    + [Example](#example-1)
      - [Transform the input/feature](#transform-the-input-feature)
      - [Transformed training data: linearly separable!](#transformed-training-data--linearly-separable-)
    + [Another Example](#another-example)
      - [How to transform the input/feature?](#how-to-transform-the-input-feature-)
      - [Transformed training data: linearly separable](#transformed-training-data--linearly-separable)
  * [Decision Boundary](#decision-boundary)
      - [Non-linear decision boundaries](#non-linear-decision-boundaries)
- [Overfitting](#overfitting)
  * [Detecting overfitting](#detecting-overfitting)
  * [Solution: Regularization](#solution--regularization)
  * [Regularized gradient descent for Linear Regression](#regularized-gradient-descent-for-linear-regression)
  * [Regularized Normal Equation](#regularized-normal-equation)
- [Regularized Logistic Regression](#regularized-logistic-regression)
- [Model Selection](#model-selection)
  * [Train/Validation/Test Sets](#train-validation-test-sets)
  * [Hyperparameter Selection](#hyperparameter-selection)
  * [N-Fold Cross Validation](#n-fold-cross-validation)
- [Bias-Variance](#bias-variance)
  * [Bias vs Variance](#bias-vs-variance)

# Recall: Linear Regression
### Hypothesis:
$h_\theta(x)=\theta_0+\theta_1x$
$\theta_i$'s: Parameters

### Cost Function
$$J(\theta_0,\theta_1)=\frac{1}{2m}\sum^m_{i=1}(h_\theta(x^{(i)})-y^{(i)})^2$$
SSD = sum of squared differences, also known as
SSE = sum of squared errors

# Multidimensional Inputs
![Pasted image 20221018150715.png|600](./attatchments/Pasted%20image%2020221018150715.png)
### Notation
$n$ = number of features
$x^{(i)}$ = input (features) of $i^{th}$ training example.
$x_j^{(i)}$ = value of feature $j$ in $i^{th}$ training example.

# Multivariate Linear Regression
### Hypothesis
$h_\theta(x)=\theta_0+\theta_1x_1+\theta_2x_2+...+\theta_nx_n$
For convenience of notation, define $x_0=1$
$\theta_i$: Parameters

### Cost Function
$$J(\theta_0,\theta_1,...,\theta_n)=\frac{1}{2m}\sum^m_{i=1}(h_\theta(x^{(i)})-y^{(i)})^2$$
### Goal
 $\underset{\theta_0,\theta_1,...,\theta_n}{minimize}\;J(\theta_0,\theta_1,...,\theta_n)$

## How? Two potential solutions
$\underset{\theta}{min}\;J(\theta;x^{(1)},y^{(1)},...,^{(m)},y^{(m)})$

### Gradient descent (or other iterative algorithm)
- Start with a guess for θ
- Change θ to decrease JJ(θ)
- Until reach minimum

### Direct minimization
- Take derivative, set to zero
- Sufficient condition for minima
- Not possible for most “interesting” cost functions

# Indirect Solution for Linear Regression
## Gradient Descent Algorithm
Set $\theta=0$
Repeat {
$\theta_j:=\theta_j-\alpha\frac{\delta}{\delta\theta_j}J(\theta)$ simultaneously for all $j=0,...,n$
} until convergence

## Gradient Descent: Intuition
![Pasted image 20221021191608.png|600](./attatchments/Pasted%20image%2020221021191608.png)
![Pasted image 20221021191633.png|600](./attatchments/Pasted%20image%2020221021191633.png)

## 2-dimensional parameters
![Pasted image 20221021191728.png|400](./attatchments/Pasted%20image%2020221021191728.png)![Pasted image 20221021191740.png|400](./attatchments/Pasted%20image%2020221021191740.png)
## Gradient Descent for Least Squares Cost
![Pasted image 20221021192115.png|600](./attatchments/Pasted%20image%2020221021192115.png)
Gradient descent computational complexity is intuitively $O(mn)$.

# Feature Normalization
- If features have very different scale, GD can get “stuck” since $x_j$ affects size of gradient in the direction of $j^{th}$ dimension
- Normalizing features to be zero-mean ($\mu$) and same-variance ($\sigma$) helps gradient descent converge faster

![Pasted image 20221022150135.png|600](./attatchments/Pasted%20image%2020221022150135.png)

# Direct Solution for Linear Regression
### Want to minimize SSD
$J(\theta_0,\theta_1,...,\theta_n)=\frac{1}{2m}\sum^m_{i=1}(h_\theta(x^{(i)})-y^{(i)})^2$

### Find minima of function
$\theta \in \mathbb{R}^{n+1}$
$\frac{\delta}{\delta\theta_j}J(\theta)=...=0$ (for every $j$)
Solve for $\theta_0,\theta_1,...,\theta_n$
![Pasted image 20221022151635.png|200](./attatchments/Pasted%20image%2020221022151635.png)

# Direct solution
### Re-write SSD using vector matrix notation
$$J(\theta_0,\theta_1,...,\theta_n)=\frac{1}{2m}\sum^m_{i=1}(h_\theta(x^{(i)})-y^{(i)})^2$$
$$=\frac{1}{2m}(X\theta-y)^T(X\theta-y)$$
### where
$$X=\begin{bmatrix}
—(x^{(1)})^T— \\
—(x^{(2)})^T—\\
\vdots \\
—(x^{(n)})^T—
\end{bmatrix}\;\;
\vec{y}=\begin{bmatrix}
—y^{(1)}— \\
—y^{(2)}—\\
\vdots \\
—(y^{(n)})^T—
\end{bmatrix}
$$

### Solution: Normal Equation
$$\theta = (X^TX)^{-1}X^Ty$$

# Derivation of Normal Equations
### SSE in matrix form
$$J(\theta)=\frac{1}{2m}(X\theta-y)^T(X\theta-y)=$$
$$=\frac{1}{2m}(\theta^T(X^TX)\theta-2(X^Ty)^T\theta+const)$$
### Take gradient with respect to $\theta$ (vector), set to 0
$$\frac{\delta J}{\delta \theta}\propto (X^TX)^{-1}X^Ty$$
$$\theta=(X^TX)^{-1}X^Ty$$

**Also known as the *least mean squares*, or *least squares* solution**
![Pasted image 20221022162452.png|600](./attatchments/Pasted%20image%2020221022162452.png)
# Trade-offs
$m$ training examples, $n$ features.

## Gradient Descent
• No need to choose $\alpha$
• Don’t need to iterate
• Works well even when $n$ is large

## Normal Equations
- No need to choose $a$
- Don't need to iterate
- Need to compute $(X^TX)^{-1}$
**Computational complexity?**
- $O(n^3)$, slow if $n$ is very large

# Maximum Likelihood Principle (ML)
## So far, we have treated outputs as noiseless
- Defined cost function as “distance to true output”
- An alternate view:
	- data $(x, y)$ are generated by unknown process
	- however, we only observe a noisy version
	- how can we model this uncertainty?
- Alternative cost function?

## How to model uncertainty in data?
![Pasted image 20221022164801.png|300](./attatchments/Pasted%20image%2020221022164801.png)
### Hypothesis
$h_\theta(x)=\theta^Tx$
$\theta$: Parameters
$D=(x^{(i)}, y^{(i)}):$ data

### New cost function
maximize probability of given model:
$p((x^{(i)}, y^{(i)})|\theta)$

## Recall: Cost Function
![Pasted image 20221022164907.png|400](./attatchments/Pasted%20image%2020221022164907.png)

## Alternative view: Maximum Likelihood
![Pasted image 20221022165006.png|400](./attatchments/Pasted%20image%2020221022165006.png)

## Maximum Likelihood: Coin Toss Example
![869C46E2-BDF2-4A6D-8715-FBFDA5AD16AB copy.jpg](869C46E2-BDF2-4A6D-8715-FBFDA5AD16AB%20copy.jpg)

## Maximum Likelihood: Normal Distribution Example
- Observe a dataset of points $D=\{x^i\}_{i=1:10}$
- Assume $x$ is generated by **Normal distribution**, $x\sim N(x|\mu, \sigma)$
- Find parameters $\theta_{ML} = [\mu, \sigma]$ that maximize $\prod^{10}_{i=1}N(x^i|\mu,\sigma)$
![Pasted image 20221022174806.png|600](./attatchments/Pasted%20image%2020221022174806.png)
## Maximum likelihood way of estimating model parameters $\theta$
- In general, assume data is generated by some distribution $$U\sim p(U|\theta)$$
- Observations (i.i.d.) $$D=\{u^{(1)}, u^{(2)},...,u^{(m)}\}$$
- Maximum likelihood estimate
![Pasted image 20221022170625.png|600](./attatchments/Pasted%20image%2020221022170625.png)

### i.i.d. Observations
- i.i.d. == **i**ndependently **i**dentically **d**istributed random variables
- If $u^i$ are i.i.d. random variables then
$$p(u^1,u^2,...,u^m)=p(u^1)p(u^2)...p(u^m)$$
- A reasonable assumption about many datasets, but not always

# Maximum Likelihood for Linear Regression
## Recall: linear regression
Observed output is the true model’s output plus noise $$t^i=h_*(x^i)+\epsilon^i$$
![Pasted image 20221022183830.png|600](./attatchments/Pasted%20image%2020221022183830.png)
$p(t|x,\theta,\beta)=N(t|h(x),\beta^{-1})$

### Probability of of one data point {$x,t$}
$p(\boldsymbol{t}|\boldsymbol{x}, \theta, \beta)= \prod^m_{i=1}N(t^{(i)}|h(x^{(i)}),\beta^{-1})$    **Likelihood Function**

### Max. likelihood solution
$\theta_{ML}=\underset{\theta}{argmax}\; p(\boldsymbol{t}|\boldsymbol{x}, \theta, \beta)$
$\beta_{ML}=\underset{\beta}{argmax}\; p(\boldsymbol{t}|\boldsymbol{x}, \theta, \beta)$

### Want to maximize
$p(\boldsymbol{t}|\boldsymbol{x}, \theta, \beta)= \prod^m_{i=1}N(t^{(i)}|h(x^{(i)}),\beta^{-1})$

### Easier to maximize log()
$$ln\;p(\boldsymbol{t}|\boldsymbol{x}, \theta, \beta)=-\frac{\beta}{2}\sum_{i=1}^m(h(x^{(i)})-t^{(i)})^2+\frac{m}{2}ln\beta-\frac{m}{2}ln(2\pi)$$
### Want to *maximize* w.r.t. $\theta$
$$ln\;p(\boldsymbol{t}|\boldsymbol{x}, \theta, \beta)=-\frac{\beta}{2}\sum_{i=1}^m(h(x^{(i)})-t^{(i)})^2+\frac{m}{2}ln\beta-\frac{m}{2}ln(2\pi)$$
### But this is same as *minimizing* sum-of-squares cost
$$\frac{1}{2m}\sum_{i=1}^m(h(x^{(i)})-t^{(i)})^2$$

### Which is the same as our SSE cost from before!!
$$J(\theta)=\frac{1}{2m}\sum_{i=1}^m(h(x^{(i)})-t^{(i)})^2$$
# Probabilistic Motivation for SSE
- Under the Gaussian noise assumption, maximizing the probability of the data points is the same as minimizing a sum-of-squares cost function
- The same as least squares method
- ML can be used for other hypotheses!
	- Note that linear regression has a closed-form solution while others may not
![Pasted image 20221022194803.png|600](./attatchments/Pasted%20image%2020221022194803.png)

# Why is $\beta$ useful for?
- Recall: we assumed observations $t$ are Gaussian given $h(x)$
- $\beta$ allows us to write down distribution over $t$, given new $x$, called **predictive distribution**
![Pasted image 20221022195253.png|400](./attatchments/Pasted%20image%2020221022195253.png)

# Predictive Distribution
Given a new input point $x$, we can now compute a distribution over the output $t$:
![79E5AFF7-0F8C-41B3-A7AD-F26AB1714528 copy.jpg|400](79E5AFF7-0F8C-41B3-A7AD-F26AB1714528%20copy.jpg)

# Non-linear Features
### Example
![Pasted image 20221022195704.png|600](./attatchments/Pasted%20image%2020221022195704.png)
$t=5e^x+2+noise$
	$h=28x-9$
change this to:
$t=5z+2+noise$
	$h=5z+1.8$

We transformed the date to make the relationship more linear by removing the exponential.

Do do this we use:

## Non-linear Basis Functions
Main idea: if the data is not linear, we can use a nonlinear mapping, or *basis function*, to transform the features to new ones
$$\phi(x):x\in R^N \rightarrow z\in R^M$$
- $M$ is the dimensionality of the new features/input $z$ (or $\phi(x)$)
- Note that $M$ could be $= N, \;>N$ or $<N$

# Polynomial basis functions
![Pasted image 20221022200116.png|600](./attatchments/Pasted%20image%2020221022200116.png)
![Pasted image 20221022200129.png|600](./attatchments/Pasted%20image%2020221022200129.png)


# Classification
## What to do if data is nonlinear?
### Example
#### Transform the input/feature
![300](attatchments/Pasted%20image%2020221023204806.png)
$\phi(x):x\in\mathbb{R}^2\to z=x_1 \cdot x_2$

#### Transformed training data: linearly separable!
![Pasted image 20221023131531.png|600](./attatchments/Pasted%20image%2020221023131531.png)

### Another Example
#### How to transform the input/feature?
![Pasted image 20221023131646.png|300](./attatchments/Pasted%20image%2020221023131646.png)
$\phi(x):x\mathbb{R}^2\to z=\begin{bmatrix}x^2\\x_1\cdot x_2\\x_2^2\end{bmatrix}$

#### Transformed training data: linearly separable
**Intuition**: suppose $\theta=\begin{bmatrix}1\\0\\1\end{bmatrix}$
Then $\theta^Tz=x_1^2+x^2_2$
i.e., the sq. distance to the origin!

## Decision Boundary
![200](attatchments/Pasted%20image%2020221023205426.png)
$h_\theta(x)=g(\theta_0+\theta_1x_1+\theta_1x_2)$
Predict "$y=1$" if $-4+x_1+x_2\geq0$

#### Non-linear decision boundaries
![200](attatchments/Pasted%20image%2020221023205538.png)
$h_\theta(x)=g(\theta_0+\theta_1x_1+\theta_2x_2+\theta_3x^2_1+\theta_4x^2_2)$
Predict "$y=1$" if $-1+x^2_1+x^2_2\geq0$

# Overfitting
![Pasted image 20221022200159.png|600](./attatchments/Pasted%20image%2020221022200159.png)

A disaster in overfitting:
![Pasted image 20221022200245.png|600](./attatchments/Pasted%20image%2020221022200245.png)

## Detecting overfitting
**Plot model complexity versus objective function on test/train data**

As model becomes more complex, performance on training keeps improving while on test data it gets worse

![Pasted image 20221022200405.png|300](./attatchments/Pasted%20image%2020221022200405.png)

**Horizontal axis:** *measure of model complexity*. In this example, we use the maximum order of the polynomial basis functions.

**Vertical axis:** For regression, it would be SSE or mean SE (MSE). For classification, the vertical axis would be classification error rate or cross-entropy error function.

## Solution: Regularization
Use regularization:
- penalize large $\theta$. How?
- add $\lambda||x||_2^2$ term to SSE cost function
- “L-2” norm squared, ie sum of sq. elements $\sum\theta^2_j$
- $\lambda$ controls amount of regularization
$$J(\theta)=\frac{1}{2m}\bigg{[}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2+\lambda\sum_{j=1}^n\theta^2_j\bigg{]}$$
## Regularized gradient descent for Linear Regression
![600](attatchments/Pasted%20image%2020221023210506.png)

## Regularized Normal Equation
![600](attatchments/Pasted%20image%2020221023210536.png)

# Regularized Logistic Regression
![600](attatchments/Pasted%20image%2020221023210618.png)

# Model Selection
![200](attatchments/Pasted%20image%2020221023210820.png)
Problem with the model above: **Overfitting!**

- fit parameters $(\theta_0,\theta_1,...,\theta_4)$ to some set of data (training set)
- how to choose regularization weight $\lambda$ and/or number of features? **“hyperparameters”**
- this is called **model selection**
- choose the model with the lowest error on the training data $J(\theta)$?

## Train/Validation/Test Sets
Solution: split data into three sets.

![200](attatchments/Pasted%20image%2020221023211219.png)

- For each value of a hyperparameter, **train** on the train set, evaluate learned parameters on the **validation** set to get $J_{VAL}$.
- Pick the model with the hyperparameter that achieved the lowest validation error $J_{VAL}$.
- Report this model’s **test** set error.

## Hyperparameter Selection
![600](attatchments/Pasted%20image%2020221023211321.png)

## N-Fold Cross Validation
- What is we don’t have enough data for train/test/validation sets?
- Solution: use N-fold cross validation.
- Split training set into train/validation sets N times
- Report average predictions over N val sets, e.g. N=10:
![600](attatchments/Pasted%20image%2020221023211405.png)

# Bias-Variance
## Bias vs Variance
![600](attatchments/Pasted%20image%2020221023211447.png)

Suppose your learning algorithm is performing less well than you were hoping. ($J_{cv}(\theta)$ or $J_{test}(\theta)$ is high.) Two likely causes: a high **bias** problem or a high **variance** problem.

![600](attatchments/Pasted%20image%2020221023211606.png)
![600](attatchments/Pasted%20image%2020221023211630.png)
![600](attatchments/Pasted%20image%2020221023211643.png)

- Understanding how different sources of error lead to bias and variance helps us improve model fitting
- Imagine you could repeat the whole model fitting process on many datasets
- Error due to Bias: The error due to bias is taken as the difference between the expected (or average) prediction of our model and the correct value which we are trying to predict
- Error due to Variance: The variance is how much the predictions for a given point vary between different realizations of the model.