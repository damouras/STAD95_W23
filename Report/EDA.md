## Exploratory Data Analysis

Describe characteristics of demand & price series

    We begin our analysis by examining the variables collected for this study: date, price, demand, and average temperature. Date is a categorical variable that represents the date of the observation, while the other variables are continuous and measured in units of dollars/MWh, MWh, and degrees Celsius, respectively.

    Looking at the summary statistics of the variables, we find that there are no missing values in the data. However, the price variable has some extreme values, with some daily prices reaching as high as 120 dollars/MWh and even negative values. These outliers could be due to factors such as sudden changes in supply or demand, or errors in the measurement or recording of the data. It will be important to examine these outliers in more detail to determine their impact on our analysis.

    Next, we examine the correlations between the variables. We find that there is a moderate positive correlation between price and demand (r = 0.48), indicating that higher demand tends to drive up the price of electricity. However, the correlation between price and temperature is weak and negative (r = -0.033), suggesting that higher temperatures do not necessarily lead to higher prices. Similarly, the correlation between temperature and demand is also weak and negative (r = -0.093), suggesting that the effect of temperature on demand is relatively small.

    To explore the seasonal patterns in the data, we plot the average daily price, demand, and temperature over time. We observe that both demand and temperature exhibit strong seasonal patterns, with peaks occurring in the summer months (June to September) and troughs in the winter months (December to February). This is likely due to the increased use of air conditioning during hot summer months and heating during cold winter months. In contrast, the price of electricity appears to be more volatile and less seasonal, with occasional spikes in price occurring throughout the year.

    Finally, we use data visualization techniques such as box plots and heatmaps to examine the relationships between the variables in more detail. These visualizations allow us to identify any non-linear relationships or interactions between the variables that may not be apparent from simple correlation analysis.

    Overall, these exploratory data analysis techniques provide us with a better understanding of the patterns and relationships in our data and help us identify potential issues or trends that may need to be addressed in our subsequent analysis.