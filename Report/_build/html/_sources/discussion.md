# Conclusion

Through this project, we have learned several important aspects of predicting electricity price and demand using various 
statistical, machine learning, and deep learning models. The results illustrate the importance of careful feature engineering, 
model selection, and the incorporation of relevant external information into the model. Besides, we have explored some 
interesting tools during writing up the report, including the <code>Jupyter Book</code> for compiling compile the report as well as
the <code>rpy2</code> library for unifying our implementation in <code>R</code> and <code>Python</code>.

We observed that our models' performance for demand prediction was quite good. However, the electricity price proved to be more
challenging to predict due to its volatile nature and the presence of extreme values. One possible approach to address this issue
would be to investigate the causes of these extreme values and incorporate this information into the models to better capture the
underlying dynamics of price fluctuations.

Another challenge we faced was the limited geographical coverage of the temperature data, which only included the Greater Toronto
Area (GTA). This made it difficult to estimate the whole of Ontario's temperature with a single column. In future work, we would
consider using a more granular temperature dataset, which includes readings from multiple locations across Ontario, to better 
capture the spatial variability of temperature and its influence on electricity demand.

To further improve the accuracy of our models, several approaches can be considered. We could incorporate additional relevant 
features into our dataset, such as other weather factors, energy demand at the sub-regional level, or additional data related to 
the electricity market, such as supply capacity, fuel costs, and regulatory factors. This would allow our models to better 
account for the various factors that influence electricity price and demand.

Additionally, for deep learning models, hyperparameter tuning could be carried out more carefully to enhance their performance.
There are also several other noteworthy models that we have not yet explored, including neural ODE (ODE-RNN, [see here](https://github.com/YuliaRubanova/latent_ode)), Bayesian matrix factorization ([see here](https://github.com/xinychen/transdim)), and particle filter. However, it is important to note that at the time of the project, there was no reliable implementation of particle filter for time series prediction.

In summary, this exercise provided valuable insights into the process of forecasting electricity price and demand. The experience
underscored the importance of model selection, feature engineering, and incorporating relevant external factors in order to 
improve forecasting accuracy. By addressing the challenges we faced and incorporating additional data sources and models, we further enhanced the accuracy of our predictions and better understood the dynamics of electricity price and demand in Ontario.
