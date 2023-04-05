# Conclusion

Describe what you learned from this exercise 

-  What things work well
-  What were the challenges
-  What can be done to further improve accuracy 

As stated in the previous part, the result illustrates the importance of careful feature engineering, model selection, and incorporation of relevant 
information into the model. We also discover that for some time series with more "random" patterns, such as the energy price, state space models can 
perform very well. This explains why some models such as exponential smoothing, Kalman filter, or dynamic factor are prefered in finance and price prediction
tasks. Besides, we have explored some interesting tool during writing up the report, including the <code>Jupyter Book</code> to combine the report as well as
the <code>rpy2</code> library that helps to unify our implementation in <code>R</code> and <code>Python</code> together.

One of the biggest challenges for this project is the computing power. We use <code>Google Colab/Jupyter Notebook</code> for all the models, which might
not be the most optimal way of implementation. Consequently, we have to reduce the data granularity from hourly scale to daily scale. 

For the deep learning models, we might tune the hyperparameters more carefully in order to improve the accuracy of these models. Otherwise, as mentioned before,
we can also include other relevant features in our dataset, such as other weather factors, or the energy demand at the sub-regional level. There are many other worth
mentioned models that we have not tried, including neural ODE (ODE-RNN, [see here](https://github.com/YuliaRubanova/latent_ode)), Bayesian matrix factorization ([see here](https://github.com/xinychen/transdim)), particle filter (Note that at the time of the project, there is no reliable implementation of particle filter for time series prediction), etc. 

(Another Version)

Through this exercise, we have learned several important aspects of predicting electricity price and demand using various 
statistical, machine learning, and deep learning models. The results illustrate the importance of careful feature engineering, 
model selection, and the incorporation of relevant external information into the model. Besides, we have explored some 
interesting tool during writing up the report, including the <code>Jupyter Book</code> to combine the report as well as
the <code>rpy2</code> library that helps to unify our implementation in <code>R</code> and <code>Python</code> together.

We observed that our models' performance for demand prediction was quite good. However, the electricity price proved to be more
challenging to predict due to its volatile nature and the presence of extreme values. One possible approach to address this issue
could be to investigate the causes of these extreme values and incorporate this information into the models to better capture the
underlying dynamics of price fluctuations.

Another challenge we faced was the limited geographical coverage of the temperature data, which only included the Greater Toronto
Area (GTA). This made it difficult to estimate the whole of Ontario's temperature with a single column. In future work, we could
consider using a more granular temperature dataset, which includes readings from multiple locations across Ontario, to better 
capture the spatial variability of temperature and its influence on electricity demand.

To further improve the accuracy of our models, several approaches can be considered. We could incorporate additional relevant 
features into our dataset, such as other weather factors, energy demand at the sub-regional level, or additional data related to 
the electricity market, such as supply capacity, fuel costs, and regulatory factors. This would allow our models to better 
account for the various factors that influence electricity price and demand.

Additionally, for deep learning models, hyperparameter tuning could be carried out more carefully to enhance their performance.
There are also several other noteworthy models that we have not yet explored, including neural ODE (ODE-RNN, [see here](https://github.com/YuliaRubanova/latent_ode)), Bayesian matrix 
factorization ([see here](https://github.com/xinychen/transdim)), and particle filter. However, it is important to note that at the time of the project, there was no reliable 
implementation of particle filter for time series prediction.

In summary, this exercise provided valuable insights into the process of forecasting electricity price and demand. The experience
underscored the importance of model selection, feature engineering, and incorporating relevant external factors in order to 
improve forecasting accuracy. By addressing the challenges we faced and incorporating additional data sources and models, we can
further enhance the accuracy of our predictions and better understand the dynamics of electricity price and demand in Ontario.
