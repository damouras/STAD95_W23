# Data Description

We analyzed electricity price, demand, and weather data in Ontario from 2020 to 2022, using variables such as hourly and daily 
electricity prices and demand, and daily average temperature data for the Greater Toronto Area (GTA). The data was collected from
reliable sources, processed, and cleaned, while considering several data limitations during the analysis.

1. Data source:
   - Hourly electricity price and demand data: Ontario's Independent Electricity System Operator (IESO) website (https://www.ieso.ca/en/Power-Data/Data-Directory)
   - Weather data: Environment and Climate Change Canada website (https://climate.weather.gc.ca)

2. Time period:
   - Data was collected from January 1, 2020 to December 31, to 2022 .
    
3. Variables:
   - Hourly electricity price: The cost of electricity in Ontario for each hour.
   - Hourly electricity demand: The amount of electricity demanded by consumers in Ontario for each hour.
   - Daily electricity price: The average of hourly prices for each day.
   - Daily electricity demand: The sum of hourly demands for each day.
   - Weather data: The average temperature in the Greater Toronto Area (GTA) for each day.
    
4. Data processing:
   - Hourly electricity price and demand data were collected from the IESO website.
   - Daily electricity prices were calculated as the average of hourly prices for each day.
   - Daily electricity demand was calculated as the sum of hourly demands for each day.
   - Weather data was collected from the Environment and Climate Change Canada website.
   - The average temperature for each day was extracted from weather stations in the GTA whose postal code started with "M".

5. Data size:
   - The data consists of daily electricity price and demand data for each hour from 2020 to 2022, resulting in 1096 data points for each variable.
   - The weather data consists of daily average temperature data for the same time period.
    
6. Data format:
   - The electricity price and demand data are presented in CSV format, with columns for date and time, price, and demand.
   - The weather data is presented in CSV format, with columns for date and average temperature.
   
7. Data limitations:
   - The data only includes electricity price and demand data for Ontario, and temperature data for the GTA, which may not be representative of other regions.
   - Other factors that may impact electricity price and demand, such as economic conditions or policy changes, are not included in the dataset.

8. Data preparation files:
   - [Prepare price and demand](../Data/1a.data_preparing.ipynb)
   - [Prepare weather data](../Data/1b.data_prep_weather.ipynb)
   - [Data cleaning and calculating](../Data/1c.process_weather_data.ipynb)
 
9. Final data file:
   - The file with the processed data is [here](../Data/final_daily.csv)
  
