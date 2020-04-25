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
saveas = 'compare-' + info_id + '-prediction-error-mean'

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

        ppc_model_mean = np.loadtxt('%s/%s-pp-only-model-mean.txt'
                % (loaddir, loadas))
        # ppc_disc_mean = np.loadtxt('%s/%s-pp-only%s-mean.txt'
        #         % (loaddir, loadas, l))
        ppc_model_error = error_measure(ppc_model_mean, data)
        # ppc_disc_error = error_measure(ppc_disc_mean, data)
        error[-1].append(ppc_model_error)

        if j == 0:
            names.append('ODE model \n& ' + discrepancy_names[i])
            names.append('ODE model\nonly')

# Tex
cmap = matplotlib.cm.get_cmap('plasma_r')
fmat = '{:<1.2f}'

counter = 1
def row(data, counter):
    colour = np.array(data) - np.min(data)
    colour /= 2. * np.max(colour)
    d = ""
    o = ""
    for i, e in enumerate(data):
        t = fmat.format(e).strip()
        c = matplotlib.colors.to_hex(cmap(colour[i]))[1:]
        d += "\definecolor{c" + str(counter) + which_model \
                + "}{HTML}{" + c + "}\n"
        o += " & \\cellcolor{c" + str(counter) + which_model \
                + "!60}" + t
        counter += 1
    return o, d, counter

tex2 = ""
tex = ""
tex += "\\begin{table*}\\centering\n"
tex += "  \\ra{1.2}\n"
tex += "  \\begin{tabularx}{\\textwidth}{@{}cccccccccc@{}}\n"
tex += "    \\arrayrulecolor{black}\\toprule\n"
tex += "    \\multicolumn{2}{c}{Model A}"
tex += " & \\multicolumn{2}{c}{Fitted with iid noise}"
tex += " & \\multicolumn{2}{c}{Fitted with GP(t)}"
tex += " & \\multicolumn{2}{c}{Fitted with GP(O, V)}"
tex += " & \\multicolumn{2}{c}{Fitted with ARMA(2, 2)}"
tex += " \\\\\n"
tex += "    \cmidrule(lr){3-4} \cmidrule(lr){5-6}"
tex += " \cmidrule(lr){7-8} \cmidrule(lr){9-10}\n"
tex += "    &"
tex += " & \\begin{tabular}{c}ODE model \\\\ \\& iid noise\end{tabular}"
tex += " & \\begin{tabular}{@{}c@{}}ODE model \\\\ only\end{tabular}"
tex += " & \\begin{tabular}{@{}c@{}}ODE model \\\\ \\& GP(t)\end{tabular}"
tex += " & \\begin{tabular}{@{}c@{}}ODE model \\\\ only\end{tabular}"
tex += " & \\begin{tabular}{@{}c@{}}ODE model \\\\ \\& GP(O, V)\end{tabular}"
tex += " & \\begin{tabular}{@{}c@{}}ODE model \\\\ only\end{tabular}"
tex += " & \\begin{tabular}{@{}c@{}}ODE model \\\\ \\& ARMA(2, 2)\end{tabular}"
tex += " & \\begin{tabular}{@{}c@{}}ODE model \\\\ only\end{tabular}"
tex += " \\\\\n"
tex += "    \\arrayrulecolor{black}\\midrule\n"
tex += "    Calibration & sinwave"
t, t2, counter = row(error[2], counter)
tex += t
tex2 += t2
tex += " \\\\\n"
tex += "    \\arrayrulecolor{black!30}\\midrule\n"
tex += "    \multirow{2}{*}{Prediction} & staircase"
t, t2, counter = row(error[1], counter)
tex += t
tex2 += t2
tex += " \\\\\n"
tex += "    & ap"
t, t2, counter = row(error[0], counter)
tex += t
tex2 += t2
tex += " \\\\\n"
tex += "    \\arrayrulecolor{black}\\bottomrule\n"
tex += "  \\end{tabularx}\n"
tex += "  \\caption{Table.}\n"
tex += "\\end{table*}"

print(tex)
print('----' * 20)
print(tex2)
