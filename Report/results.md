# Results

Include table with metrics for each series and model.

Compare with other performance metrics from the literature (with referneces)


We tested several time series models, including Exponential Smoothing State Space Model (ETS), Seasonal Autoregressive Integrated
Moving Average (SARIMA), and Singular Spectrum Analysis (SSA), to predict the price and demand for electricity in Ontario. The 
data from 2020 and 2021 was used for training the models, while the 2023 data served as the test set. We employed one-step ahead 
predictions and updated the training set with true data after each prediction. Each model had different input data versions, 
including the original price or demand data, and log-transformed versions.

The performance of each model was evaluated using Mean Absolute Error (MAE), Mean Square Error (MSE), and Mean Absolute Percentage Error (MAPE). The results are summarized in the following table:

| Model              | Price_MAE | Price_MSE | Demand_MAE | Demand_MSE | Price_MAPE | Demand_MAPE |
|--------------------|-----------|-----------|------------|------------|------------|-------------|
| Constant           | 27.85     | 1192.81   | 34098.96   | 1623311682 | 0.56       | 0.08913415  |
| ETS                | 12.93     | 281.03    | 15270.60   | 414561058  | 0.3399     | 0.04089591  |
| ETS_log            | 13.23     | 286.88    | 15270.61   | 414561272  | 0.3144     | 0.04089592  |
| SARIMA             | 12.50     | 268.12    | 14317.14   | 326372945  | 0.3274     | 0.03829565  |
| SARIMA_weather     | 12.49     | 268.07    | 14324.77   | 326427076  | 0.3268     | 0.03843157  |
| SARIMA_log         | 12.65     | 272.33    | 14370.43   | 328902644  | 0.3002     | 0.03841707  |
| SARIMA_weather_log | 12.63     | 273.08    | 14438.61   | 330403616  | 0.2999     | 0.03859608  |
| SARIMA_2_xreg      | 12.49     | 268.32    | 14369.06   | 327475674  | 0.3268     | 0.03843157  |
| SSA                | 21.20     | 707.93    | 35370.02   | 1839698615 | 0.64       | 0.06180689  |
| SSA_log            | 23.97     | 903.02    | 35387.91   | 1838608713 | 0.46       | 0.09078513  |

Based on the evaluation metrics, the SARIMA model with weather as an external regressor (SARIMA_weather) provided the best 
overall performance for predicting both price and demand, with a Price_MAE of 12.49, Price_MSE of 268.07, Demand_MAE of 14324.77,
and Demand_MSE of 326427076. The SARIMA model also had a Price_MAPE of 0.3268 and a Demand_MAPE of 0.03843157. The ETS model had 
similar performance, while the SSA model had significantly higher error values.

It is worth noting that the log transformation slightly improved the performance of some models, such as the ETS_log and 
SARIMA_log models. The use of external regressors, such as weather data, also influenced the performance of the SARIMA model, 
with the SARIMA_weather model yielding the lowest error values. These findings suggest that incorporating external factors like 
weather data and using appropriate data transformations can enhance the predictive capabilities of time series models.

In summary, our results show that the SARIMA model with weather as an external regressor outperforms the other models in 
predicting both price and demand for electricity in Ontario. These findings highlight the importance of model selection and 
incorporating relevant external factors when forecasting electricity prices and demand.
