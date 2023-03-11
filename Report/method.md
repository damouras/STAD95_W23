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

## Exponential smoothing
The prediction of the exponential smoothing model can be expressed as:

$$\hat{y}_{t+1|t} = \alpha y_t + \alpha (1 - \alpha) y_{t-1} + \alpha (1 - \alpha)^2 y_{t-2} + ... $$

where $0 \leq \alpha \leq 1$ is the smoothing parameter. We can also write the forecast at time $t + 1$ as a weighted average between the most recent observation $y_t$ and the previous forecast $\hat{y}_{t|t-1}$:

$$\hat{y}_{t+1|t} = \alpha y_t + (1 - \alpha) \hat{y}_{t|t-1}$$

## Kalman filter
## Dynamic factor
## XGBoost
## Gaussian process
## Fast Fourier transform
## Singular spectrum analysis
## Long short-term memory (LSTM)
## Transformer 
## Structured state space model (S4)

