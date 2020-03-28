import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from datetime import date, timedelta
import matplotlib.animation as animation

# Look you're on a branch! WAHEY!

plt.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\bin\ffmpeg.exe'

data = pd.read_csv("ie_cases_130320.csv")
x_range = 40
predict_st = 13
start_date = '2020-02-29'
sd = start_date.split('-')
days = pd.DataFrame(list(range(0, x_range)))
dates = pd.to_datetime(list(range(0, x_range)), unit='D', origin=pd.Timestamp(start_date))
data['Date'] = pd.to_datetime(list(range(0, len(data['Days']))), unit='D', origin=pd.Timestamp(start_date))

def sigmoid(x, L, x0, k, b):
    b = 1
    return L / (1 + np.exp(-k * (x - x0))) + b
def r_squared(y, y_fit):
    residuals = y - y_fit
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2
def round_sf(number, significant):
    return round(number, significant - len(str(number)))

fig = plt.figure()
ax = plt.axes(xlim=(date(int(sd[0]), int(sd[1]), int(sd[2])), date(int(sd[0]), int(sd[1]), int(sd[2])) + timedelta(x_range)), ylim=(0, 2500))
plt.xticks(rotation=90)
plt.title('Projected No. of Cases of COVID-19 in Ireland.')

line, = ax.plot([], [])
sc, = ax.plot([], [], marker='o', ls='', color = 'blue')

p0 = [10000, 25.5, 0.4, 1]
params = []
y_fits = []
fit_exs = []
max_cases = []
max_date = []
for i in range(predict_st, len(data['Cases']) + 1):
    param, param_cov = curve_fit(sigmoid, data['Days'][:i], data['Cases'][:i], p0, method='dogbox', maxfev=1000000)
    y_fit = param[0] / (1 + np.exp(-param[2] * (data['Days'][:i] - param[1]))) + param[3]
    fit_ex = (param[0] / (1 + np.exp(-param[2] * (days - param[1]))) + param[3]).values
    params.append(param)
    y_fits.append(y_fit)
    fit_exs.append(fit_ex)
    r2 = r_squared(data['Cases'][:i], y_fit)
    max_day = round(param[1] - np.log(param[3] / (param[0] - param[3])) / param[2])
    max_date.append(date(int(sd[0]), int(sd[1]), int(sd[2])) + timedelta(max_day))
    max_cases.append(round_sf(int(param[0]),2))

line_anim = []
j = -1

for i in range((100 * (len(fit_exs)-1))+1):
    if i % 100 == 0:
        j += 1
        line_anim.append(fit_exs[j])
    else:
        fit_ex = []
        for k in range(len(fit_exs[j])):
            fit_ex.append(fit_exs[j][k] + (i % 100) * (fit_exs[j + 1][k] - fit_exs[j][k]) / 100)
        line_anim.append(fit_ex)

def init():
    sc.set_data([], [])
    line.set_data([], [])
    return [sc, line]

text_box = ax.text(0.02, 0.9, '', transform=ax.transAxes)
maxmax_cases = max(np.log(max_cases))
cmap = plt.get_cmap("RdYlGn")

j = 0
def animate(i):
    global j
    if (i - 1) <= predict_st:
        sc.set_data(dates[:i], data['Cases'][:i])
        line.set_data([], [])
        j = i
        text_box.set_text('Max cases of ____\nby ____')
    else:
        if i + 1 >= len(line_anim) + predict_st:
            sc.set_data(dates[:j], data['Cases'][:j])
            line.set_data(dates, line_anim[-1])
            line.set_color(cmap(1 - np.log(max_cases[j - predict_st - 1]) / maxmax_cases))
            text_box.set_text(
                'Max cases of {}\n by {}  '.format(max_cases[-1], max_date[-1]))
        else:
            if (i - predict_st) % 100 == 0:
                j += 1
                line.set_color(cmap(1 - np.log(max_cases[j - predict_st - 1]) / maxmax_cases))
            else:
                color_val_curr = np.log(max_cases[j - predict_st - 1]) / maxmax_cases
                color_val_next = np.log(max_cases[j - predict_st]) / maxmax_cases
                assign_color = color_val_curr + (color_val_next - color_val_curr)*((i - predict_st) % 100)/100
                line.set_color(cmap(1 - assign_color))
            sc.set_data(dates[:j], data['Cases'][:j])
            line.set_data(dates, line_anim[i - predict_st])
            text_box.set_text('Max cases of {}\n by {}  '.format(max_cases[j - predict_st - 1],max_date[j - predict_st - 1]))
    return tuple([sc, line]) + tuple([text_box])


Writer = animation.writers['ffmpeg']
writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=1800)
anim = animation.FuncAnimation(fig, animate, len(line_anim)+predict_st+300, init_func=init, interval=1, blit=False)
anim.save('animation7.mp4', writer=writer)

plt.show()
