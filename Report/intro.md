# Introduction

```{tableofcontents}
```

Price: predicting next few hours
Demand: longer time frame

Source: [Independent Electricity System Operator (IESO)](https://www.ieso.ca/en/)
(add description)
- final_data.csv (n = 9144)
    - Date (9144)
    - Price (9144)
    - 1_hour_pred (9127)
    - 2_hour_pred (9127)
    - 3_hour_pred (9127)
    - Market Demand (9144)
    - Ontario Demand (9144)
    - 1_lag_pred (9126): prediction from one hour ago
    - 2_lag_pred (9125): prediction from two hour ago
    - 3_lag_pred (9124): prediction from three hour ago

- 5minutes.csv (n = 840)
    - Price
    - Demand
    - Supply
- PUB_demand_202*.csv (n = 8761)
    - Date
    - Hour
    - Market Demand
    - Ontario Demand
- hourlay_data_202*.csv (n = 8761)
    - Date
    - Hour
    - HOEP
    - Hour 1 predispatch
    - Hour 2 predispatch
    - Hour 3 predispatch
    - OR 10 min sync
    - OR 10 min non-sync
    - OR 30 min
- generation_fuel_type_multiday.xml (n = 140)
    - Supply_BIOFUEL
    - Supply_GAS
    - Supply_HYDRO
    - Supply_NUCLEAR
    - Supply_SOLAR
    - Supply_WIND
- ontario_demand_multiday.xml
    - Demand_5_Minute (n = 1729)
    - Demand_Actual (n = 140)
    - Demand_Projected (n = 140)
- price_multiday.xml 
    - Price_HOEP (n = 140): Hourly Ontario energy price
    - Price_HOEP_Projected (n = 140): Projected hourly Ontario energy price
    - Price_MCP (n = 1729): Market clearing price
