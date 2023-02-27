
library(Rssa)

df = read.csv('final_daily.csv')

price_test = tail(df["price"], 365)
price_train = head(df["price"], length(ts(df['price'])) - 365)

s <- ssa(log(price_train+5))
for1 <- rforecast(s, groups = list(trend = c(1:12)), len = 365, recurrent=TRUE)

for1 = exp(for1)
plot(ts(price_test+5))
lines(ts(for1), col='blue')

mean((ts(for1) - ts(price_test+5))**2)

mean(abs(ts(for1) - ts(price_test+5)))

##Demand

demand_test = tail(df["demand"], 365)
demand_train = head(df["demand"], length(ts(df['demand'])) - 365)

s <- ssa(demand_train)
for1 <- rforecast(s, groups = list(trend = c(1:12)), len = 365, recurrent=TRUE)

plot(ts(demand_test))
lines(ts(for1), col='blue')

mean((ts(for1) - ts(demand_test))**2)

mean(abs(ts(for1) - ts(demand_test)))
