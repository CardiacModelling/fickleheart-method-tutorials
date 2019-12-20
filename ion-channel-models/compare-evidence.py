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
predict_list = ['Sinewave', 'Staircase', 'AP'][::-1]
discrepancy_list = ['', '-gp', '-gp-ov', '-arma_2_2']
load_list = ['-iid', '-gp', '-gp', '-armax']
discrepancy_names = ['iid noise', 'GP(t)', 'GP(O, V)', 'ARMA(2, 2)']

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
saveas = 'compare-' + info_id + '-evidence'

error = []
names = []
for j, which_predict in enumerate(predict_list):
    error.append([])

    for i, (d, l) in enumerate(zip(discrepancy_list, load_list)):
        # Load mean error
        loaddir = './fig/mcmc-' + info_id + d
        loadas = info_id + '-sinewave-' + which_predict

        if j == 0:
            names.append(discrepancy_names[i])

        ppc_error = np.loadtxt('%s/%s-posterior.txt' % (loaddir, loadas))
        error[-1].append(ppc_error)

# Plot
cmap = matplotlib.cm.get_cmap('plasma_r')
fmat = '{:<1.2f}'

targs = {
    'horizontalalignment': 'center',
    'verticalalignment': 'center',
}

def row(axes, y, data, std=None):
    data = np.copy(data)
    data = data - np.max(data)
    colour = np.array(data) - np.min(data)
    colour /= 2. * np.max(colour)
    colour = np.abs(0.5 - colour)
    for i, e in enumerate(data):
        x = 4 * i
        w, h = 4, 1
        r = plt.Rectangle(
	    (x, y), w, h, facecolor=cmap(colour[i]), alpha=0.6)
        ax.add_patch(r)

        text = fmat.format(e).strip()
        if std:
            text += ' (' + fmat.format(std[i]).strip() + ')'
        plt.text(x + w / 2., y + h / 2., text, **targs)

plt.figure(figsize=(5.2, 1.8))
plt.subplots_adjust(0.005, 0.005, 0.995, 0.995)

xperbox = 4.
plt.xlim(-6.2, xperbox * 4)
plt.ylim(0, 4)

ax = plt.subplot(1, 1, 1)
ax.set_xticks([])
ax.set_yticks([])
for i, n in enumerate(names):
    plt.text(xperbox / 2. + xperbox * i, 3.5, n, **targs)
#plt.text(xperbox * 2, 4.5, 'Fitted with GP(t)', **targs)
#plt.plot([xperbox + 0.25, xperbox + xperbox * 2 - 0.25], [4.2]*2, c='#dddddd')
#plt.text(xperbox * 4, 4.5, 'Fitted with GP(O, V)', **targs)
#plt.plot([xperbox * 3 + 0.25, xperbox * 3 + xperbox * 2 - 0.25], [4.2]*2,
#        c='#dddddd')
#plt.text(xperbox * 6, 4.5, 'Fitted with ARMA(2, 2)', **targs)
#plt.plot([xperbox * 5 + 0.25, xperbox * 5 + xperbox * 2 - 0.25], [4.2]*2,
#        c='#dddddd')

label = 'Model %s' % which_model
plt.text(-2, 3.5, label, {'weight': 'normal', 'size': 14}, **targs)

for i, n in enumerate(predict_list):
    plt.text(-1.5, i + .5, n, **targs, fontsize=9)
plt.text(-4.5, 2.5, 'Calibration', **targs, fontsize=9)
plt.plot([-2.9]*2, [2.1, 2.9], c='#dddddd')
plt.text(-4.5, 1.0, 'Prediction', **targs)
plt.plot([-2.9]*2, [0.2, 1.8], c='#dddddd')

for i, e in enumerate(error):
    row(ax, i, e)

plt.axvline(0, color='#dddddd')
plt.axhline(2, color='#dddddd')
plt.axhline(5, color='#555555')
plt.axhline(3, color='#555555')

plt.savefig(savedir + '/' + saveas + '.pdf', format='pdf')
plt.savefig(savedir + '/' + saveas + '.png', dpi=300)
plt.close()

