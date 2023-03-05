raw = read.csv("final_daily.csv")
library(xts)
library(lubridate)
library(vars)
library(forecast)
library(astsa)
library(tidyverse)
date_time = as_datetime(raw[,1])
price = xts( x = raw[,"price"], order.by = date_time) 
demand = xts( x = raw[,"demand"], order.by = date_time) 

demand_train = demand["2020/2021"]; demand_test = demand["2022"]
price_train = price["2020/2021"]; price_test = price["2022"]
df_fc = cbind(price_test,demand_test)
#### Demand ####

library(forecast)
library(tictoc)
# tic("auto.arima")
for (i in 1:365){
  demand = rbind(demand_train,head(demand_test,i))
  price = rbind(price_train,head(price_test,i))
  X. = cbind(demand, price)
  out = VAR(X., ic = "AIC", lag.max = 24) %>% restrict()
  forecast = predict(out,n.ahead = 1)
  d_fore = forecast$fcst$demand[1]
  p_fore = forecast$fcst$price[1]
  df_fc[i,"demand_test"] = d_fore[1]
  df_fc[i,"price_test"] = p_fore[1]
}


# plot(demand_train); lines(demand_fitd, col = 2)

# For 1-step-ahead ARIMA predictions using same model while updating  data
# see: https://stats.stackexchange.com/questions/55168/one-step-ahead-forecast-with-new-data-collected-sequentially
demand = ts( c(demand_train, demand_test))
price = ts( c(price_train, price_test))
X=cbind( demand, price)
vars::causality( out, cause = "demand_train" )

newfit = VAR( X, model = out)

library(magrittr)
demand_pred = tail( as.numeric(newfit$fitted), n = length(demand_test) ) %>%
  xts(x = ., order.by = time(demand_test))

plot(demand_test); lines(demand_pred, col = 2)

# In-sample 
mean( ( demand_train - demand_fitd )^2 )
mean( abs( demand_train - demand_fitd ) )

# Out-of-sample MAE
perf[2,1] = mean( abs( demand_test - demand_pred ) )
perf[2,2] = mean( ( demand_test - demand_pred )^2 )



#### Price  ####

out = auto.arima( ts(price_train), seasonal = TRUE, xreg = ts(demand_train))
## ARIMA(1,1,3)(0,0,2)[24] 
price_fitd = xts( as.numeric(out$fitted), order.by = time(price_train) )

newfit = Arima( ts( c(price_train, price_test)), model = out, xreg=ts( c(demand_train, demand_test)))

price_pred = tail( as.numeric(newfit$fitted), n = length(price_test) ) %>%
  xts(x = ., order.by = time(price_test))

plot(price_test); lines(price_pred, col = 2)

# In-sample 
mean( ( price_train - price_fitd )^2 )
mean( abs( price_train - price_fitd ) )

# Out-of-sample MAE
perf[2,3] = mean( abs( price_test - price_pred ) )
perf[2,4] = mean( ( price_test - price_pred )^2 )
perf