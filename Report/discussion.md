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
