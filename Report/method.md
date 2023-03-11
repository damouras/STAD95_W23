# Data preprocessing

# Overview of the model

## SARIMA

SARIMA is the abbreviation for Seasonal Autoregressive Integrated Moving Average model. To introduce the SARIMA model, we will give some overview about the canonical autoregressive (AR) model and moving average (MA) model.

An autoregressive of order $p$, denotes AR($p$), can be written as:

$$y_t = c + \phi_1 y_{t-1} +\phi_2 y_{t-2} + ... + \phi_p y_{t - p} + \epsilon_t$$

where $\epsilon_t$ is the white noise.

Rather 

## Vector autoregressive (VAR)
## Regression with ARIMA errors
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

