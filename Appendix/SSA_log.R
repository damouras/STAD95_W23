library(Rssa)

df = read.csv('final_daily.csv')

price_test = tail(df["price"], 365)
price_train = head(df["price"], length(ts(df['price'])) - 365)

price_train = log(price_train+5)
price_test = price_test+5

s <- ssa(price_train, L = 100)
for1 <- rforecast(s, groups = list(trend = c(1:24)), len = 365, recurrent=TRUE)

for1=exp(for1)

plot(ts(price_test))
lines(ts(for1), col='blue')

mean((ts(for1) - ts(price_test))**2)

mean(abs(ts(for1) - ts(price_test)))

error = abs(ts(price_test)-ts(for1))/abs(ts(price_test))
error[error > 1] = 1
mean(error)

##Demand

demand_test = tail(df["demand"], 365)
demand_train = head(df["demand"], length(ts(df['demand'])) - 365)

demand_train=log(demand_train)
demand_test=log(demand_test)

s <- ssa(demand_train, L = 100)
for1 <- rforecast(s, groups = list(trend = c(1:24)), len = 365, recurrent=TRUE)

demand_test = exp(demand_test)
for1=exp(for1)
plot(ts(demand_test))
lines(ts(for1), col='blue')

mean((ts(for1) - ts(demand_test))**2)

mean(abs(ts(for1) - ts(demand_test)))

error = abs(ts(demand_test)-ts(for1))/abs(ts(demand_test))
error[error > 1] = 1
mean(error)