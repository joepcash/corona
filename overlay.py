import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from datetime import date, timedelta

data = pd.read_csv("ie_cases_130320.csv")
x_range = 28
predict_st = 20
days = pd.DataFrame(list(range(0,x_range)))
dates = pd.to_datetime(list(range(0,x_range)), unit='D', origin=pd.Timestamp('2020-02-29'))
data['Date'] =  pd.to_datetime(list(range(0,len(data['Days']))), unit='D', origin=pd.Timestamp('2020-02-29'))

def sigmoid(x, L ,x0, k, b):
    return L / (1 + np.exp(-k*(x-x0))) + b

def r_squared(y, y_fit):
    residuals = y - y_fit
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

cmap = plt.get_cmap("autumn")
#colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
p0 = [10000, 25.5, 0.4, 1]
pr_dates = []
params = []
y_fits = []
fit_exs = []
for i in range(predict_st,len(data['Cases'])):
    param, param_cov = curve_fit(sigmoid, data['Days'][:i], data['Cases'][:i], p0, method='dogbox', maxfev=1000000)
    y_fit = param[0] / (1 + np.exp(-param[2]*(data['Days'][:i]-param[1])))+param[3]
    fit_ex = param[0] / (1 + np.exp(-param[2]*(days-param[1])))+param[3]
    params.append(param)
    y_fits.append(y_fit)
    fit_exs.append(fit_ex)
    pr_dates.append(date(2020,2,29) + timedelta(i))
    r2 = r_squared(data['Cases'][:i],y_fit)
    max_day = round(param[1] - np.log(param[3]/(param[0] - param[3]))/param[2])
    max_date = date(2020,2,29) + timedelta(max_day)
    print(dates[i],param[0],max_date)

data.plot(kind='scatter', x='Date', y='Cases', color='green')
for i in range(predict_st,len(data['Cases'])):
    plt.plot(dates, fit_exs[i-predict_st], color=cmap((i-predict_st)/(len(fit_exs)-1)))
    #print((i-14)/(len(fit_exs)-1))
#plt.plot(dates, fit_exs[0], color='green', label = '$14^{th}$ Mar (reaches max cases of 43,000,000 by $13^{th}$ Jun)')
#plt.plot(dates, fit_exs[1], color='blue', label = '$16^{th}$ Mar (reaches max cases of 1,182 by $8^{th}$ Apr)')
#plt.plot(dates, fit_exs[2], color='orange', label = '$18^{th}$ Mar (reaches max cases of 107,000,000 by $16^{th}$ Jun)')
#plt.plot(dates, fit_exs[3], color='purple', label = '$20^{th}$ Mar (reaches max cases of 6,368 by $25^{th}$ Apr)')
#plt.plot(dates, fit_exs[4], color='skyblue', label = '$21^{st}$ Mar (reaches max cases of 1,506 by $4^{th}$ Apr)')
#plt.legend(loc="upper left")
plt.title('Projected No. of Cases of COVID-19 in Ireland from date.')
ax = plt.axes()
ax.set_facecolor("lightgrey")
plt.xticks(rotation=90)
plt.ylim((0,2000))
plt.show()