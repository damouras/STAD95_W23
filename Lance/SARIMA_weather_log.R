raw = read.csv("final_daily.csv")
library(xts)
library(lubridate)

date_time = as_datetime(raw[,1])
price = xts( x = raw[,"price"], order.by = date_time) 
demand = xts( x = raw[,"demand"], order.by = date_time) 
temp = xts( x = raw[,"avg_temp"], order.by = date_time)

demand_train = demand["2020/2021"]; demand_test = demand["2022"]
price_train = price["2020/2021"]; price_test = price["2022"]
temp_train = temp["2020/2021"]; temp_test = temp["2022"]


perf = matrix(0, nrow = 3, ncol = 4 )
rownames(perf) = c("Constant", "SARIMA", "EWMA")
colnames(perf) = c("Demand | MAE", "Demand | MSE", "Price | MAE", "Price|MSE")

perf[1,1] = mean( abs( demand_test - mean(demand_train) ) )
perf[1,2] = mean( ( demand_test - mean(demand_train) )^2 )
perf[1,3] = mean( abs( price_test - mean(price_train) ) )
perf[1,4] = mean( ( price_test - mean(price_train) )^2 )

#### Demand ####

library(forecast)
library(tictoc)
# tic("auto.arima")
demand_train = log(demand_train)
demand_test = log(demand_test)
out = auto.arima( ts(demand_train), seasonal = TRUE, xreg = ts(temp_train) )
## ARIMA(2,0,0)(2,1,0)[24] with drift 
# toc() 
## auto.arima: 109.06 sec elapsed ~ 2min
demand_fitd = xts( as.numeric(out$fitted), order.by = time(demand_train) )

# plot(demand_train); lines(demand_fitd, col = 2)

# For 1-step-ahead ARIMA predictions using same model while updating  data
# see: https://stats.stackexchange.com/questions/55168/one-step-ahead-forecast-with-new-data-collected-sequentially
newfit = Arima( ts( c(demand_train, demand_test)),xreg=ts( c(temp_train, temp_test)), model = out)

library(magrittr)
demand_pred = tail( as.numeric(newfit$fitted), n = length(demand_test) ) %>%
  xts(x = ., order.by = time(demand_test))

demand_test=exp(demand_test)
demand_pred=exp(demand_pred)
plot(demand_test); lines(demand_pred, col = 2)

# In-sample 
mean( ( demand_train - demand_fitd )^2 )
mean( abs( demand_train - demand_fitd ) )

# Out-of-sample MAE
perf[2,1] = mean( abs( demand_test - demand_pred ) )
perf[2,2] = mean( ( demand_test - demand_pred )^2 )



#### Price  ####
price_train=log(price_train+5)
price_test=log(price_test+5)
out = auto.arima( ts(price_train), seasonal = TRUE, xreg = ts(temp_train))
## ARIMA(1,1,3)(0,0,2)[24] 
price_fitd = xts( as.numeric(out$fitted), order.by = time(price_train) )

newfit = Arima( ts( c(price_train, price_test)), model = out, xreg=ts( c(temp_train, temp_test)))

price_pred = tail( as.numeric(newfit$fitted), n = length(price_test) ) %>%
  xts(x = ., order.by = time(price_test))

price_pred=exp(price_pred)
price_test=exp(price_test)

plot(price_test); lines(price_pred, col = 2)

# In-sample 
mean( ( price_train - price_fitd )^2 )
mean( abs( price_train - price_fitd ) )

# Out-of-sample MAE
perf[2,3] = mean( abs( price_test - price_pred ) )
perf[2,4] = mean( ( price_test - price_pred )^2 )
perf