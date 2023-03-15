# Data preprocessing

# Overview of the model

## SARIMA

SARIMA is the abbreviation for Seasonal Autoregressive Integrated Moving Average model. To introduce the SARIMA model, we will give some overview about the canonical autoregressive (AR) model and moving average (MA) model.

An autoregressive of order $p$, denotes AR($p$), can be written as:

$$y_t = c + \phi_1 y_{t-1} +\phi_2 y_{t-2} + ... + \phi_p y_{t - p} + \epsilon_t$$

where $\epsilon_t$ is the white noise. This is like a multiple regression but with lagged values of $y_t$ as the predictors.

Rather than using the past values of $y_t$ in a regression, a moving average model uses past forecast errors in a regression-like model:

$$y_t = c + \epsilon_t + \theta_1 \epsilon_{t - 1} + \theta_2 \epsilon_{t-2} + ... + \theta_q \epsilon_{t - q}$$

We refer to this model as an MA($q$) model, a moving average model of order $q$.

## Vector autoregressive (VAR)
## Regression with ARIMA errors

Often, when we use linear regression (with respect to time), we consider regression models of the form:

$$y_t = \beta_0 + \beta_1 x_{1, t} + ... + \beta_k x_{k, t} + \epsilon_t$$

where $y_t$ is a linear function of the $k$ predictor variables $(x_{1, t},..., x_{k, t})$, and $\epsilon_t$ is usually assumed to be an uncorrelated error term (i.e white noise). 

For time series, we can also allow the errors from a regression to contain autocorrelation. Instead of using $\epsilon_t$, we can use $\eta_t$. The error series $\eta_t$ is assumed to follow some ARIMA models:

$$y_t = \beta_0 + \beta_1 x_{1, t} + ... + \beta_k x_{k, t} + \eta_t$$

For example, if $\eta_t \sim$ ARIMA(1, 1, 1), then we can write:

$$(1 - \phi_1 B)(1 - B)\eta_t = (1 + \theta_1 B)\epsilon_t$$

Here, $B$ denotes the backshift operator, and $\epsilon_t$ is a white noise series.

## Exponential smoothing
The prediction of the exponential smoothing model can be expressed as:

$$\hat{y}_{t+1|t} = \alpha y_t + \alpha (1 - \alpha) y_{t-1} + \alpha (1 - \alpha)^2 y_{t-2} + ... $$

where $0 \leq \alpha \leq 1$ is the smoothing parameter. We can also write the forecast at time $t + 1$ as a weighted average between the most recent observation $y_t$ and the previous forecast $\hat{y}_{t|t-1}$:

$$\hat{y}_{t+1|t} = \alpha y_t + (1 - \alpha) \hat{y}_{t|t-1}$$

## Kalman filter

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

$$\mathbf{S}_k = \mathbf{H}_k \mathbf{P}^{-}_k + \mathbf{R}_k$$
$$\mathbf{K}_k = \mathbf{P}^{-}_k \mathbf{H}^{\top})_k \mathbf{S}_k^{-1}$$
$$\mathbf{m}_k = \mathbf{m}^{-}_k + \mathbf{K}_k (\mathbf{y}_k - \mathbf{H}_k \mathbf{m}^{-}_k)]$$

## Dynamic factor
## XGBoost

XGBoost is a supervised learning algorithm that can be used for both regression and classification. It attempts to predict the target variable by combining the estimates of a set of simpler and weaker models. 

When using gradient boosting for regression, the weak learners are regression trees, and each regression tree maps an input data point to one of its leafs that contains a continuous score. XGBoost minimizes an objective function that combines a convex loss function (based on the difference between the predicted and target outputs) and a penalty term for model complexity (in other words, the regression tree functions). The training proceeds iteratively, adding new trees that predict the residuals or errors of prior trees. These trees are then combined with previous trees to make the final prediction. It's called gradient boosting because it uses a gradient descent algorithm to minimize the loss when adding new models.

![url-to-image](https://docs.aws.amazon.com/images/sagemaker/latest/dg/images/xgboost_illustration.png)


## Gaussian process
## Fast Fourier transform
## Singular spectrum analysis
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

