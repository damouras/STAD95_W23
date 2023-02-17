# STAD95
Exploring non-linear time series techniques

## Feb 13 (week 6)

### Lance
- [ ] Aggregate to daily data (sum of demands, avg of prices) 
- [ ] Combine daily weather to previous data (all in one master daily file)
- [ ] Update: constant/SARIMA/ETS models for daily data (Log-transform prices)
- [ ] Try models with external regressors
- [ ] Try Gaussian Process model

### Nghia
- [ ] Refit following models on daily data
- Dynamic factor
- Kalman Filter 
- Transformer 
- (LSTM ?)
- XGboost
- [ ] Try models with external regressors
- [ ] Try SSA (Singular Spectrum Analysis) on daily data

#####
Table for filling in model performance

| Model          | Price | Demand |
|----------------|-------|--------|
| Simple Mean    |       |        |
| IESO forecasts |       |        |
| ETS            |       |        |
| SARIMA         |       |        |
| XGBoost        |       |        |
| GP             |       |        |
| Dynamic Factor |       |        |
| Kalman Filter  |       |        |
| Transformer    |       |        |
| FFT            |       |        |
| SSA            |       |        |
|                |       |        |
|                |       |        |
|                |       |        |
|                |       |        |
|                |       |        |

## Feb 6 (week 5)

### Summer
- [X] Try auto.arima in R (tried this but price need to be transformed)
- [ ] Try a different package in python
- [X] Test out transformations
- [ ] Set up report sceleton 

### Lance
- [ ] Iterative 1-step ahead predictions for Exponential Weighting 
    (try forecats::ets() in R first, using process similar to  Arima for getting test set 1-step-ahead predictions: 
    ```
    > out =  ets( train )
    > fit = ets( c(train, test), model = out )
    > pred = tail( fitted(fit ), n = length(test) )
    ```
- [ ] Choose memory parameter
- [ ] Weather data

### Nghia
- [ ] LSTM
- [ ] Dynamic factor
- [ ] Explain how to create 1-step-ahead predictions *incorporating latest test data* for nonlinear auto-regression in Python (RegressorChain)

### General Comments

- Test all models on 2022 data
- If using transforms, convert predictions back to original scale before calculating MSE
- Tried constant and SARIMA model for Demand & (untransformed) Price in R (code in NOTES/SARIMA.R).
  The 1-step-ahed forecasts for the test data were caclulated by fitting the model on the train data 
  to the entire (train + test) data, and extracting the fitted values for the latter test part
  (see also https://stats.stackexchange.com/questions/217955/difference-between-first-one-step-ahead-forecast-and-first-forecast-from-fitted)
  The resulting test data performance is:

|         |      Demand - MAE|      Demand - MSE|      Price - MAE|    Price - MSE|
|:--------|-----------------:|-----------------:|----------------:|--------------:|
|Constant |         1925.5107|        5623541.59|         31.83118|      2003.9185|
|SARIMA   |          173.2907|          51301.12|         14.08960|       703.8321|


## Jan 30 (week 4)

### Notes
- Fit 1D models to forecast 2022 Price & Demand, based on 2020-21 values. 
- Use rolling 1-step-ahead predictions, and report Mean Square Error (MSE) and Mean Absolute Error (MAPE).
- Reminders:
  1. Use personal branch when making changes
  2. Keep track of what we've done / major changes made (under the list of tasks)
  3. Message the group when making changes that affect everybody (i.e. moving files / changing folder structure). 
  4. Make sure the Jupyter Book builds successfully before we commit and push.

### All
- [ ] Improve documentation (intro)

### Summer  
- [X] next monday 5pm? let everyone know by Wed
- [X] SARIMA
- [X] explore transformations for price data

### Lance
- [x] save preprocessed data to load into each notebook
- [ ] [Exponential smoothing](https://www.statsmodels.org/dev/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html)
- [ ] State space models
- [ ] Check off last week's tasks

### Nghia
- [ ] Neural nets
- [ ] Check off last week's tasks (also add XGboost & Gaussian process to jupyter book?)

## Jan 27 (week 3)

### Summer
- [x] create jupyter book
  - [x] insert files, format toc
  - [x] insert outline for intro
- [x] fit vanilla ARMA

Additional changes
- changed absolute path to relative path
- renamed columns (snake case)

### Lance
- [X] Insert work into Jupyter Book
- [ ] Fill out documentation

### Nghia
- [x] Insert work into Jupyter Book
- [x] Fill out documentation

Additional changes
- XGboost
- Gaussian process regression

## Git Best Practices

0. Check that you are on the correct branch using `git status`
1. If your branch is behind main, check out your branch and do `git merge main`, then push
2. Once you finish your work and push to your branch, your branch is ahead of main. Check out main and do `git merge yourbranch`, then push

## Resources

[Jupyter book docs](https://jupyterbook.org/en/stable/basics/organize.html)

Relevant notebooks in python
  - https://github.com/Carterbouley/ElectricityPricePrediction
  - https://github.com/KuanWeiBeCool/Predict-Electricity-Demand-in-Ontario
  - https://github.com/SLTaiwo/CIND820-Forecasting/blob/main/CIND820Projectv1.ipynb
  - https://github.com/Philippe-Beaudoin/Electricity-Storage-Thesis-Ontario

Cool visualizations in R
  - https://github.com/nathankchan/ontario-electricity-demand-viz#list-of-visualizations
