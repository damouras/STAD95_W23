# SARIMA
# Vector autoregressive (VAR)
# Regression with ARIMA errors
# Exponential smoothing
The prediction of the exponential smoothing model can be expressed as:

$$\hat{y}_{T+1|T} = \alpha y_T + \alpha (1 - \alpha) y_{T-1} + \alpha (1 - \alpha)^2 y_{T-2} + ... $$

where $0 \leq \alpha \leq 1$ is the smoothing parameter. We can also write the forecast at time $T + 1$ as a weighted average between the most recent observation $y_T$ and the previous forecast $\hat{y}_{T|T-1}$:

$$\hat{y}_{T+1|T} = \alpha y_T + (1 - \alpha) \hat{y}_{T|T-1}$$

# Kalman filter
# Dynamic factor
# XGBoost
# Gaussian process
# Fast Fourier transform
# Singular spectrum analysis
# Long short-term memory (LSTM)
# Transformer 
# Structured state space model (S4)

