def animate_cases_prediction_history(full_range_days,
                                     prediction_start_day, start_date, country,
                                     params_initial_guess = [10000, 25.5, 0.4, 1]):
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    import numpy as np
    from datetime import date, timedelta
    import matplotlib.animation as animation

    # Look you're on a branch! WAHEY!

    # Define path to ffmpeg for saving mp4
    plt.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\bin\ffmpeg.exe'

    data = pd.read_csv("ie_cases_130320.csv")
    start_date_ymd = start_date.split('-')
    days = pd.DataFrame(list(range(0, full_range_days)))
    dates = pd.to_datetime(list(range(0, full_range_days)),
                           unit='D', origin=pd.Timestamp(start_date))
    data['Date'] = pd.to_datetime(list(range(0, len(data['Days']))),
                                  unit='D', origin=pd.Timestamp(start_date))

    fig = plt.figure()
    ax = plt.axes(xlim=(date(int(start_date_ymd[0]),
                             int(start_date_ymd[1]), int(start_date_ymd[2])),
                        date(int(start_date_ymd[0]), int(start_date_ymd[1]),
                             int(start_date_ymd[2])) + timedelta(full_range_days)),
                  ylim=(0, 5000))
    plt.xticks(rotation=90)
    plt.title('Projected No. of Cases of COVID-19 in ' + country + '.')
    line, = ax.plot([], [])
    sc, = ax.plot([], [], marker='o', ls='', color = 'blue')

    # Continue cleaning up code from here
    params = []
    y_fits = []
    fit_exs = []
    max_cases = []
    max_date = []
    for i in range(prediction_start_day, len(data['Cases']) + 1):
        param, param_cov = curve_fit(sigmoid, data['Days'][:i], data['Cases'][:i], params_initial_guess, method='dogbox', maxfev=1000000)
        y_fit = param[0] / (1 + np.exp(-param[2] * (data['Days'][:i] - param[1]))) + param[3]
        fit_ex = (param[0] / (1 + np.exp(-param[2] * (days - param[1]))) + param[3]).values
        params.append(param)
        y_fits.append(y_fit)
        fit_exs.append(fit_ex)
        r2 = r_squared(data['Cases'][:i], y_fit)
        max_day = round(param[1] - np.log(param[3] / (param[0] - param[3])) / param[2])
        max_date.append(date(int(start_date_ymd[0]), int(start_date_ymd[1]), int(start_date_ymd[2])) + timedelta(max_day))
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
        if (i - 1) <= prediction_start_day:
            sc.set_data(dates[:i], data['Cases'][:i])
            line.set_data([], [])
            j = i
            text_box.set_text('Max cases of ____\nby ____')
        else:
            if i + 1 >= len(line_anim) + prediction_start_day:
                sc.set_data(dates[:j], data['Cases'][:j])
                line.set_data(dates, line_anim[-1])
                line.set_color(cmap(1 - np.log(max_cases[j - prediction_start_day - 1]) / maxmax_cases))
                text_box.set_text(
                    'Max cases of {}\n by {}  '.format(max_cases[-1], max_date[-1]))
            else:
                if (i - prediction_start_day) % 100 == 0:
                    j += 1
                    line.set_color(cmap(1 - np.log(max_cases[j - prediction_start_day - 1]) / maxmax_cases))
                else:
                    color_val_curr = np.log(max_cases[j - prediction_start_day - 1]) / maxmax_cases
                    color_val_next = np.log(max_cases[j - prediction_start_day]) / maxmax_cases
                    assign_color = color_val_curr + (color_val_next - color_val_curr)*((i - prediction_start_day) % 100)/100
                    line.set_color(cmap(1 - assign_color))
                sc.set_data(dates[:j], data['Cases'][:j])
                line.set_data(dates, line_anim[i - prediction_start_day])
                text_box.set_text('Max cases of {}\n by {}  '.format(max_cases[j - prediction_start_day - 1],max_date[j - prediction_start_day - 1]))
        return tuple([sc, line]) + tuple([text_box])


    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=1800)
    anim = animation.FuncAnimation(fig, animate, len(line_anim)+prediction_start_day+300, init_func=init, interval=1, blit=False)
    anim.save('animation9.mp4', writer=writer)

    plt.show()
def sigmoid(x, L, x0, k, b):
    import numpy as np
    b = 1
    return L / (1 + np.exp(-k * (x - x0))) + b
def r_squared(y, y_fit):
    import numpy as np
    residuals = y - y_fit
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2
def round_sf(number, significant):
    return round(number, significant - len(str(number)))

if __name__ == "__main__":
    animate_cases_prediction_history(50,13,'2020-02-29', 'Ireland')