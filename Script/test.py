import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.rcParams['figure.dpi'] = 300
from pylab import rcParams
import seaborn as sns
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARMA, ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import scipy.stats as stats
from pmdarima import auto_arima
import datetime

data = pd.read_csv('../Data/final_data.csv')
df = data[['date', 'price', '1_hour_pred', '2_hour_pred', '3_hour_pred', 'ontario_demand']]
df = df.set_index('date')

plt.figure(figsize=(10,6))
sns.heatmap(df.isna().transpose(),
            cmap="YlGnBu",
            cbar_kws={'label': 'Missing Data'});

y = df.ontario_demand
train = y[y.index < '2022']
valid = y[(y.index >= '2022') & (y.index < '2023')]

model = auto_arima(train, start_p=0, start_q=0, seasonal=True, m = 24, maxiter = 10, method = 'nm', stepwise = True)
model.fit(train)
print(model)