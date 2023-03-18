# Data preprocessing

# Overview of the model

## SARIMA

The implement of SARIMA in <code>Python</code> is supported by the <code>pmdarima</code> as well as the <code>statsmodels</code> package. In <code>R</code>, the model is implemented using the <code>forecast</code> package.

SARIMA is the abbreviation for Seasonal Autoregressive Integrated Moving Average model. To introduce the SARIMA model, we will give some overview about the canonical autoregressive (AR) model and moving average (MA) model.

An autoregressive of order $p$, denotes AR($p$), can be written as:

$$y_t = c + \phi_1 y_{t-1} +\phi_2 y_{t-2} + ... + \phi_p y_{t - p} + \epsilon_t$$

where $\epsilon_t$ is the white noise. This is like a multiple regression but with lagged values of $y_t$ as the predictors.

Rather than using the past values of $y_t$ in a regression, a moving average model uses past forecast errors in a regression-like model:

$$y_t = c + \epsilon_t + \theta_1 \epsilon_{t - 1} + \theta_2 \epsilon_{t-2} + ... + \theta_q \epsilon_{t - q}$$

We refer to this model as an MA($q$) model, a moving average model of order $q$.

## Vector autoregressive (VAR)

The implement of VAR in <code>Python</code> is supported by the <code>statsmodels</code> package. In <code>R</code>, the model is implemented using the <code>vars</code> package

## Regression with ARIMA errors

The implement of regression with ARIMA errors in <code>R</code> is supported by the <code>forecast</code> package.

Often, when we use linear regression (with respect to time), we consider regression models of the form:

$$y_t = \beta_0 + \beta_1 x_{1, t} + ... + \beta_k x_{k, t} + \epsilon_t$$

where $y_t$ is a linear function of the $k$ predictor variables $(x_{1, t},..., x_{k, t})$, and $\epsilon_t$ is usually assumed to be an uncorrelated error term (i.e white noise). 

For time series, we can also allow the errors from a regression to contain autocorrelation. Instead of using $\epsilon_t$, we can use $\eta_t$. The error series $\eta_t$ is assumed to follow some ARIMA models:

$$y_t = \beta_0 + \beta_1 x_{1, t} + ... + \beta_k x_{k, t} + \eta_t$$

For example, if $\eta_t \sim$ ARIMA(1, 1, 1), then we can write:

$$(1 - \phi_1 B)(1 - B)\eta_t = (1 + \theta_1 B)\epsilon_t$$

Here, $B$ denotes the backshift operator, and $\epsilon_t$ is a white noise series.

## Exponential smoothing
The implement of exponential smoothing in <code>Python</code> is supported by the <code>statsmodels</code> package. In <code>R</code>, the model is implemented using the <code>forecast</code> package using the <code>ets</code> function.

The prediction of the exponential smoothing model can be expressed as:

$$\hat{y}_{t+1|t} = \alpha y_t + \alpha (1 - \alpha) y_{t-1} + \alpha (1 - \alpha)^2 y_{t-2} + ... $$

where $0 \leq \alpha \leq 1$ is the smoothing parameter. We can also write the forecast at time $t + 1$ as a weighted average between the most recent observation $y_t$ and the previous forecast $\hat{y}_{t|t-1}$:

$$\hat{y}_{t+1|t} = \alpha y_t + (1 - \alpha) \hat{y}_{t|t-1}$$

## Kalman filter

We implement two version of Kalman filter in <code>Python</code>. The simple version is an implementation from scratch using <code>Numpy</code>, and the advanced version using the class <code>KalmanForecaster</code> from the <code>darts</code> package.

Let's define $\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_T$ to be the states and $\mathbf{y}_1, \mathbf{y}_2, ..., \mathbf{y}_T$ to be the measurements. Generally, state space models have the following form:

$$\mathbf{x}_k \sim p(\mathbf{x_k} | \mathbf{x}_{k-1})$$

$$\mathbf{y}_k \sim p(\mathbf{y}_k | \mathbf{x}_k)$$

$$\mathbf{x}_0 \sim p(\mathbf{x}_0)$$

for $k = 1, 2, ..., T$. The first expression is called the dynamic model, which represents the dynamic of the states. The second one is called the measurement model, capturing the measurements and their uncertainties. The last expression is called the prior distribuion, which contains the information about the state before obtaining any measurements.

Our goal is to recursively compute those marginal distributions:
- Filtering distribution: $p(\mathbf{x}_k|\mathbf{y}_1, ..., \mathbf{y}_k)$
- Prediction distribution: $p(\mathbf{x}_{k + n}|\mathbf{y}_1, ..., \mathbf{y}_k)$, for $n = 1, 2, ...$

We will define our linear Gaussian state space model (Kalman filter) in the same structure that we define the state space model above. Specifically, we define:

$$p(\mathbf{x}_k|\mathbf{x}_{k - 1}) = N(\mathbf{x}_k | \mathbf{A}_{k - 1}\mathbf{x}_{k -1}, \mathbf{Q}_{k - 1})$$

$$p(\mathbf{y}_k|\mathbf{x}_k) = N(\mathbf{y}_k | \mathbf{H}_k \mathbf{x}_k, \mathbf{R}_k)$$

$$p(\mathbf{x}_0) = N(\mathbf{x}_0 | \mathbf{m}_0, \mathbf{P}_0)$$

The Kalman filter actually calculates the following distributions:

$$p(\mathbf{x}_k|\mathbf{y}_{1:k-1}) = N(\mathbf{x}_k | \mathbf{m}^{-}_k, \mathbf{P}^{-}_k)$$

$$p(\mathbf{x}_k|\mathbf{y}_{1:k}) = N(\mathbf{x}_k | \mathbf{m}_k, \mathbf{P}_k)$$

The **prediction step** of Kalman filter:

$$\mathbf{m}_k^{-} = \mathbf{A}_{k - 1}\mathbf{m}_{k - 1}$$

$$\mathbf{P}_k^{-} = \mathbf{A}_{k - 1}\mathbf{P}_{k - 1}\mathbf{A}^{\top}_{k - 1} + Q_{k - 1}$$

The **update step** of Kalman filter:

$$\mathbf{S}_k = \mathbf{H}_k \mathbf{P}^{-}_k \mathbf{H}^{\top}_k+ \mathbf{R}_k$$

$$\mathbf{K}_k = \mathbf{P}^{-}_k \mathbf{H}^{\top}_k \mathbf{S}_k^{-1}$$

$$\mathbf{m}_k = \mathbf{m}^{-}_k + \mathbf{K}_k (\mathbf{y}_k - \mathbf{H}_k \mathbf{m}^{-}_k)$$

$$\mathbf{P}_k = \mathbf{P}^{-}_k - \mathbf{K}_k \mathbf{S}_k \mathbf{K}_k^{\top}$$

## Dynamic factor

The implement of the dynamic factor model in <code>Python</code> is supported by the <code>statsmodels</code> package.

The dynamic factor can be written as:

$$y_t = \mu_t + \mathbf{\Lambda_t} \mathbf{f_t} + e_t$$

where $\mathbf{f_t}$ is the static factor vector, $e_t$ is the idiosyncratic disturbances, and $\mathbf{\Lambda_t}$ is the factor loading matrix. Each factor in the dynamic factor model (i.e each $f_{jt}$, where $j = 1, ..., q$) is modelled as an autoregressive stationary process.

## XGBoost

The implement of the XGBoost model for time series regression in <code>Python</code> is supported by the <code>xgboost</code> package, together with the <code>RegressorChain</code> module from the <code>scikit-learn</code> package.

XGBoost is a supervised learning algorithm that can be used for both regression and classification. It attempts to predict the target variable by combining the estimates of a set of simpler and weaker models. 

When using gradient boosting for regression, the weak learners are regression trees, and each regression tree maps an input data point to one of its leafs that contains a continuous score. XGBoost minimizes an objective function that combines a convex loss function (based on the difference between the predicted and target outputs) and a penalty term for model complexity (in other words, the regression tree functions). The training proceeds iteratively, adding new trees that predict the residuals or errors of prior trees. These trees are then combined with previous trees to make the final prediction. It's called gradient boosting because it uses a gradient descent algorithm to minimize the loss when adding new models.

![url-to-image](https://docs.aws.amazon.com/images/sagemaker/latest/dg/images/xgboost_illustration.png)


## Gaussian process

The implement of the Gaussian process regression model for time series in <code>Python</code> is supported by the <code>sklearn</code> package, together with the <code>RegressorChain</code> module also from <code>scikit-learn</code>.

A Gaussian process is a probability distribution over functions $\hat{y}(\mathbf{x})$ such that for any set of values of $\hat{y}(\mathbf{x})$ evaluated at an arbitrary set of points $\mathbf{x}^{(1)}, \mathbf{x}^{(2)}, ...$ is jointly Gaussian. Recall that a linear model can be written in a form:

$$y | \mathbf{x} \sim N(\hat{y}(\mathbf{x}), \sigma^2)$$

$$\hat{y}(\mathbf{x}) = \mathbf{w}^{\top} \psi(\mathbf{x})$$

where $\phi(\mathbf{x}): \mathbb{R}^D \rightarrow \mathbb{R}^M$ is the feature map. In the multivariate case, where $N$ samples are given, we can write:

$$\mathbf{y} | \hat{\mathbf{y}} \sim N(\hat{\mathbf{y}}, \sigma^2 I)$$

Since $\hat{\mathbf{y}}$ is a (zero-mean) Gaussian process, we have: $\hat{\mathbf{y}} \sim N(0, \mathbf{K})$. Therefore, the marginal of $\mathbf{y}$ is given by:

$$\mathbf{y} \sim N(0, \mathbf{C})$$

$$\mathbf{C} = \mathbf{K} + \sigma^2 I$$

with each elements of the covariance matrix $\mathbf{C}$ defined as: 

$$C(\mathbf{x}^{(i)}, \mathbf{x}^{(j)}) = \frac{1}{\alpha}k(\mathbf{x}^{(i)}, \mathbf{x}^{(j)}) + \sigma^2 \delta_{ij}$$

where $k(\mathbf{x}, \mathbf{x}^{'}) = \psi(\mathbf{x})^{\top} \psi(\mathbf{x}^{'})$ be the positive semidefinite kernel defined by the feature maps.

To find the predictive distribution, or the distribution for the new output $y^{N + 1}$ given its past, we can use the conditional distribution of the Multivariate Normal distribution. Specifically, we have:

$$p(y^{(N + 1)} | \mathbf{y}_N) = N(\mathbf{k}^{\top} \mathbf{C}_N^{-1} \mathbf{y}_N, c - \mathbf{k}^{\top} \mathbf{C}_N^{-1} \mathbf{k})$$

where:

$$\mathbf{y}_{N + 1} \sim N(0, \mathbf{C}_{N + 1})$$

$$\mathbf{C}_{N + 1} = \mathbf{K}_{N + 1} + \sigma^2 I$$

$$C_{N + 1}(\mathbf{x}^{(i)}, \mathbf{x}^{(j)}) = \frac{1}{\alpha} k(\mathbf{x}^{(i)}, \mathbf{x}^{(j)}) + \sigma^2 \delta_{ij}$$

$$c = \frac{1}{\alpha} k(\mathbf{x}^{(N + 1)}, \mathbf{x}^{(N + 1)}) + \sigma^2$$

Note that the vector $\mathbf{k}$ is a vector with entries $k_i = \frac{1}{\alpha} k(\mathbf{x}^{(i)}, \mathbf{x}^{(N + 1)})$. Hence, $\mathbf{k}$ is a function of the new test input $\mathbf{x}^{(N + 1)}$.


## Fast Fourier transform

The implement of FFT for time series prediction in <code>Python</code> is from the <code>darts</code> package.

Let $x_1, x_2, ..., x_N$ be a sequence of length $N$. We define the Fast Fourier transform $y_k$ of length $N$ as:

$$y_k = \sum^{N - 1}_{n = 0} e^{-2 \pi j \frac{kn}{N}} x_n$$

The inverse FFT is defined as follow:

$$x_n = \frac{1}{N} \sum^{N - 1}_{k = 0} e^{2 \pi j \frac{kn}{N}} y_k$$

With the assumption that the time series is periodic, we can use the FFT to extrapolate the time series, which is equivalent to making prediction. The <code>darts</code> package allows us to choose how many frequencies to keep in order to forecast the time series.

## Singular spectrum analysis

The model is implemented using the <code>ssa</code> package in <code>R</code>.

We consider a time series $\mathbf{y_T} = (y_1, ..., y_T)$. Fix $L$ such that $L < \frac{T}{2}$, the window length, and let $K = T - L + 1$. 

First, we need to compute the trajectory matrix $\mathbf{X} = [X_1, ..., X_K]$ as follow:

```math
\mathbf{X} = (x_{ij})_{i, j = 1}^{L, K} = \begin{bmatrix}
y_1 & y_2 & y_3 & y_4 &\ldots & y_{K} \\ 
y_2 & y_3 & y_4 & y_5 &\ldots & y_{K + 1} \\
y_3 & y_4 & y_5 & y_6 &\ldots & y_{K + 2} \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\
y_{L} & y_{L + 1} & y_{L + 2} & y_{L + 3} & \ldots & y_{T} \\ 
\end{bmatrix}
```

Note that the trajectory matrix $\mathbf{X}$ is a Hankel matrix, which means that all the elements along the diagonal $i + j = $const are equal. 

Then, we apply SVD (Singular Value Decomposition) for the matrix $\mathbf{X}\mathbf{X}^{\top}$. To calculate the SVD, we need to calculate the eigenvalues and eigenvectors of the matrix $\mathbf{X}\mathbf{X}^{\top}$ and represent it in the form $\mathbf{X}\mathbf{X}^{\top} = \mathbf{P}\mathbf{\Lambda}\mathbf{P}^{\top}$. Here, $\mathbf{\Lambda} = diag(\lambda_1, ..., \lambda_L)$ is the diagonal matrix of $\mathbf{X}\mathbf{X}^{\top}$ ordered such that $\lambda_1 \geq \lambda_2 \geq...\geq \lambda_L \geq 0$ and $\mathbf{P} = (P_1, ..., P_L)$ is the corresponding orthogonal matrix of eigenvectors of $\mathbf{X}\mathbf{X}^{\top}$.

Next, we will reconstruct the time series from the SVD that we obtain above. We would divide the set of indices $\{1, ..., L\}$ into m disjoint partitions $I_1, ..., I_m$. Let $I = \{i_1, ..., i_p\}$. Then the matrix $\mathbf{X_I}$ corresponding to group $I$ is defined as $\mathbf{X}_I = \mathbf{X}_{i_1} + ... + \mathbf{X_{i_p}}$. Computing this quantity for each group $I_1, ..., I_m$ as constructed above, we have:

$$\mathbf{X} = \mathbf{X}_{I_1} + ... + \mathbf{X}_{I_m}$$

## Long short-term memory (LSTM)

Long short-term memory (LSTM) is a type of neural network that is explicitly designed to avoid the long-term dependency problem. LSTM uses three gates (input, forget and output gates) to control the flow of information. Each gate is implemented as a sigmoid layer that receives the input and the previous hidden state, and produces a value between 0 and 1. The update equations are as follows:

$$ i_t = \sigma(W_{xi} \cdot x_t + W_{hi} \cdot h_{t-1} + b_i)$$

$$ f_t = \sigma(W_{xf} \cdot x_t + W_{hf} \cdot h_{t-1} + b_f)$$

$$ o_t = \sigma(W_{xo} \cdot x_t + W_{ho} \cdot h_{t-1} + b_o)$$

$$ c_t = f_t \cdot c_{t-1} + i_t \cdot \tanh(W_{xc} \cdot x_t + W_{hc} \cdot h_{t-1} + b_c) $$

$$ h_t = o_t \cdot \tanh(c_t)$$

where $i_t$ is the input gate, $o_t$ is the output gate, $f_t$ is the forget gate, $c_t$ is the memory cell, and $h_t$ is the hidden state. We denote $\sigma$ as the sigmoid function, which is defined as $\sigma(x) = \frac{1}{1 + e^{-x}}$.


## Transformer 

Transformer is a type of neural network architecture that is used for sequential data, such as NLP tasks or time series data. The model is known for its ability to efficiently handle long-term dependencies and parallelizable computation. The underlying core of Transformer model is the **self-attention mechanism**, which allows the model to weigh the importance of different parts of the input when making predictions. Furthermore, the model has an encoder-decoder architecture, where the encoder is responsible for processing the input sequence and the decoder is mainly responsible for producing the output sequence.

The attention mechanism can be mathematically represented as:
$$Attention(Q, K, V) = softmax(\frac{QK^{\top}}{\sqrt{}d_k})V$$

where $Q$, $K$, and $V$ are matrices representing the query, key, and value respectively. $d_k$ is the dimension of the key.

The attention mechanism is applied multiple times in the Transformer model, in a multi-head attention mechanism, where multiple sets of queries, keys, and values are used. The output of the multi-head attention is then concatenated and passed through a linear layer.



## Structured state space model (S4)

