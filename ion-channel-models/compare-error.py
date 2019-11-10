#!/usr/bin/env python3
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def rmse(t1, t2):
    # Root mean square error
    return np.sqrt(np.mean(np.power(np.subtract(t1, t2), 2)))

error_measure = rmse

model_list = ['A', 'B']
predict_list = ['sinewave', 'staircase', 'ap'][::-1]
discrepancy_list = ['', '-gp', '-arma_2_2']
load_list = ['', '-gp', '-armax']
discrepancy_names = ['iid noise', 'GP(t)', 'ARMAX(2, 2)']

try:
    which_model = sys.argv[1] 
except:
    print('Usage: python %s [str:which_model]' % os.path.basename(__file__))
    sys.exit()

if which_model not in model_list:
    raise ValueError('Input model %s is not available in the model list' \
            % which_model)
info_id = 'model_%s' % which_model

savedir = './fig/compare'
if not os.path.isdir(savedir):
    os.makedirs(savedir)
saveas = 'compare-' + info_id + '-prediction-error'

error = []
names = []
for j, which_predict in enumerate(predict_list):
    # Load data
    data_dir = './data'
    data_file_name = 'data-%s.csv' % which_predict
    data = np.loadtxt(data_dir + '/' + data_file_name,
                      delimiter=',', skiprows=1)  # headers
    times = data[:, 0]
    data = data[:, 1]

    error.append([])

    for i, (d, l) in enumerate(zip(discrepancy_list, load_list)):
        # Load predictions
        loaddir = './fig/mcmc-' + info_id + d + '/raw'
        loadas = info_id + '-sinewave-' + which_predict
        times_predict = np.loadtxt('%s/%s-pp-time.txt' % (loaddir, loadas))
        assert(all(np.abs(times - times_predict) < 1e-6))

        ppc_mean = np.loadtxt('%s/%s-pp%s-mean.txt' % (loaddir, loadas, l))
        ppc_error = error_measure(ppc_mean, data)
        error[-1].append(ppc_error)
        if j == 0:
            names.append(discrepancy_names[i])

        if d == '':
            # ppc_model_mean = np.NaN
            # ppc_disc_mean = np.NaN
            pass
        else:
            ppc_model_mean = np.loadtxt('%s/%s-pp-only-model-mean.txt'
                    % (loaddir, loadas))
            # ppc_disc_mean = np.loadtxt('%s/%s-pp-only%s-mean.txt'
            #         % (loaddir, loadas, l))
            ppc_model_error = error_measure(ppc_model_mean, data)
            # ppc_disc_error = error_measure(ppc_disc_mean, data)
            error[-1].append(ppc_model_error)
            if j == 0:
                names.append('Model with\n' + discrepancy_names[i])

# Plot
cmap = matplotlib.cm.get_cmap('plasma_r')
fmat = '{:<1.2f}'

targs = {
    'horizontalalignment': 'center',
    'verticalalignment': 'center',
}

def row(axes, y, data, std=None):
    colour = np.array(data) - np.min(data)
    colour /= 2. * np.max(colour)
    for i, e in enumerate(data):
        x = 4 * i
        w, h = 4, 1
        r = plt.Rectangle(
	    (x, y), w, h, facecolor=cmap(colour[i]), alpha=0.6)
        ax.add_patch(r)

        text = fmat.format(e).strip()
        if std:
            text += ' (' + fmat.format(std[i]).strip() + ')'
        plt.text(x + w / 2, y + h / 2, text, **targs)

plt.figure(figsize=(8, 1.5))
plt.subplots_adjust(0.005, 0.005, 0.995, 0.995)

plt.xlim(-4.2, 20)
plt.ylim(0, 4)

ax = plt.subplot(1, 1, 1)
ax.set_xticks([])
ax.set_yticks([])
for i, n in enumerate(names):
    plt.text(2 + 4 * i, 3.5, n, **targs)

label = 'Model A'
plt.text(-1.75, 3.5, label, {'weight': 'normal', 'size': 14}, **targs)

for i, n in enumerate(predict_list):
    plt.text(-2.1, i + .5, n, **targs)

for i, e in enumerate(error):
    row(ax, i, e)

plt.axvline(0, color='#dddddd')
plt.axhline(4, color='#555555')
plt.axhline(3, color='#555555')

plt.savefig(savedir + '/' + saveas + '.pdf', format='pdf')
plt.savefig(savedir + '/' + saveas + '.png', dpi=300)
plt.close()

