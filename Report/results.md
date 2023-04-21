# Results

We tested several time series models, including Exponential Smoothing State Space Model (ETS), Seasonal Autoregressive Integrated
Moving Average (SARIMA), and Singular Spectrum Analysis (SSA), to predict the price and demand for electricity in Ontario. The 
data from 2020 and 2021 was used for training the models, while the 2023 data served as the test set. We employed one-step ahead 
predictions and updated the training set with true data after each prediction. Each model had different input data versions, 
including the original price or demand data, and log-transformed versions.

The performance of each model was evaluated using Mean Absolute Error (MAE), Mean Square Error (MSE), and Mean Absolute Percentage Error (MAPE). The results are summarized in the following table:

| Model                  | Price_MAE | Price_MSE | Price_MAPE | Demand_MAE | Demand_MSE    | Demand_MAPE   |
|------------------------|-----------|-----------|------------|------------|---------------|---------------|
| Constant               |     27.85 |   1192.81 |       0.56 |   34098.96 | 1623311682.00 |    0.08913415 |
| ETS                    |     12.93 |    281.03 |     0.3399 |   15270.60 |  414561058.00 |    0.04089591 |
| ETS_log                |     13.23 |    286.88 |     0.3144 |   15270.61 |  414561272.00 |    0.04089592 |
| SARIMA                 |     12.50 |    268.12 |     0.3274 |   14317.14 |  326372945.00 |    0.03829565 |
| SARIMA_weather         |     12.49 |    268.07 |     0.3268 |   14324.77 |  326427076.00 |    0.03843157 |
| SARIMA_log             |     12.65 |    272.33 |     0.3002 |   14370.43 |  328902644.00 |    0.03841707 |
| SARIMA_weather_log     |     12.63 |    273.08 |     0.2999 |   14438.61 |  330403616.00 |    0.03859608 |
| SARIMA_2_xreg          |     12.49 |    268.32 |     0.3268 |   14369.06 |  327475674.00 |    0.03843157 |
| SSA                    |     15.76 |    412.85 |       0.41 |   19292.29 |  676312825.00 |    0.05126683 |
| SSA_log                |     18.71 |    598.25 |       0.45 |   19373.10 |  693156155.00 |    0.05137282 |
| VAR                    |     23.64 |    898.23 |       0.33 |   42462.15 | 2750605858.94 | 0.03999313156 |
| Kalman filter (simple) |     12.94 |    280.92 |       0.34 |   19236.02 | 1108128912.54 | 0.05357406444 |
| Kalman forecaster      |     13.08 |    286.66 |       0.34 |   17641.76 |  512782421.67 | 0.04739837037 |
| Dynamic factor         |     13.31 |    301.65 |       0.38 |   14874.44 |  388028400.16 | 0.04001733041 |
| Transformer            |     19.28 |    600.66 |       0.57 |   17509.55 |  489325704.16 | 0.04769337162 |
| ETS (Python)           |     12.95 |    281.01 |       0.33 |   15270.60 |  414561050.30 | 0.04100255746 |
| Fast Fourier Transform |     32.71 |   1527.91 |       0.91 |   29343.31 | 1240550013.47 | 0.09738732874 |
| XGBoost                |     17.93 |    568.41 |       0.47 |   15339.24 |  419687148.36 | 0.04041825596 |
| Gaussian Process       |     12.84 |    274.91 |       0.34 |   14334.84 |  329325202.30 | 0.03825991559 |
| LSTM                   |     22.72 |    800.78 |       0.67 |   23636.19 |  845069403.35 |  0.0633824304 |
| Structure State Space  |     30.04 |   1342.67 |       0.85 |   34469.24 | 1741656924.36 | 0.09769348272 |

Based on the evaluation metrics, the SARIMA model with weather as an external regressor (SARIMA_weather) provided the best 
overall performance for predicting both price and demand, with a Price_MAE of 12.49, Price_MSE of 268.07, Demand_MAE of 14324.77,
and Demand_MSE of 326427076. The SARIMA model also had a Price_MAPE of 0.3268 and a Demand_MAPE of 0.03843157. The ETS model had 
similar performance, while the SSA model had significantly higher error values.

It is worth noting that the log transformation slightly improved the performance of some models, such as the ETS_log and 
SARIMA_log models. The use of external regressors, such as weather data, also influenced the performance of the SARIMA model, 
with the SARIMA_weather model yielding the lowest error values. These findings suggest that incorporating external factors like 
weather data and using appropriate data transformations can enhance the predictive capabilities of time series models.

On the other hand, among the machine learning and deep learning models, Gaussian process performs very well and almost matches the
performance of the SARIMA model. Other state space models, such as both versions of Kalman filter and the dynamic factor model, also have very low error which approaches that of SARIMA. The neural network models, such as LSTM, Transformer,
and structured state space model (S4) do not work as well as other machine learning models or statistical models in this case. However, it is
noticeable that as we add more series to the model (such as combining the demand and average temperature series together with the price series), 
the error of these models (MSE, MAE, MAPE) is reduced. One implication is that deep learning models might perform well when the dataset
contains many different series, from which the model can use its large capacity to learn the correlation between these series and make 
predictions accordingly.

In summary, our results show that the SARIMA model with weather as an external regressor outperforms the other models in 
predicting both price and demand for electricity in Ontario. These findings highlight the importance of model selection and 
incorporating relevant external factors when forecasting electricity prices and demand.
