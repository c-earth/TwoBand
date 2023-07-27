import os
import numpy as np
from scipy.optimize import curve_fit

from utils.processing import read_file, plot_prediction, plot_residual, plot_fitting_params, plot_sigma, plot_relative_residual
from utils.model import *


# absolute path to data folder
data_dir = 'D:/python_project/TwoBand/S2/'
resu_dir = data_dir[:-1] + '_two_band_fitting/'
if not (os.path.exists(resu_dir)):
    os.makedirs(resu_dir)

# analysed range of field in Tesla (recommend to use the whole range)
Hmin = 0.0
Hmax = 9.0

# scale conductance so that all fitting parameters are around 1
S_scale = 1E6
q = 1.602E-19

# put in the temperatures that the fitting are bad, and want to omit from the plots.
T_omit = []

# model controller, change True/False for whether the data will be fitted to the model.
models = [['1a', False, model1a, model1a_verify, model1a_post_param], 
          ['1b', False, model1b, model1b_verify, model1b_post_param], 
          ['2', False, model2, model2_verify, model2_post_param], 
          ['3', False, model3, model3_verify, model3_post_param], 
          ['4', True, model4, model4_verify, model4_post_param]]

# load data
data = dict()
for f in os.listdir(data_dir):
    term = f[6: 8]
    T = int(f[9:f.find('K')])
    data[T] = data.get(T, dict())
    data[T][term] = read_file(data_dir + f, Hmin, Hmax)    

# start fitting at different temperature
print('Start fitting the data at the following temperature.')
Temperatures = np.sort(list(data.keys()))
print('T(K): ',Temperatures)
Bxxs = []
Sxxs = []
Bxys = []
Sxys = []

params_files = []
for i, T in enumerate(Temperatures):
    if T in T_omit:
        continue

    print('\n')
    print('--------------------------------------')
    print(f'T = {T} K')
    print('--------------------------------------')

    Bxx, Sxx = data[T]['xx']
    Bxxs.append(Bxx)
    Sxx /= S_scale
    Sxxs.append(Sxx)
    Bxy, Sxy = data[T]['xy']
    Bxys.append(Bxy)
    Sxy /= S_scale
    Sxys.append(Sxy)
    Bs = np.concatenate((Bxx, Bxy))
    Ss = np.concatenate((Sxx, Sxy))
    BsSs = np.concatenate((Bs, Ss))

    # initial guess on fitting parameters
    uh0 = 1
    nh0 = 1
    ue0 = 1
    ne0 = 1
    sxx0 = Sxx[0]

    p0 = [uh0, nh0, ue0, ne0, Sxx[0]]

    # adaptive fitting guess, since we know that the fitting should be closed to the previous temperature fitted.
    ps = {'1a':p0.copy(), '1b':p0.copy(), '2':p0.copy(), '3':p0.copy(), '4':p0.copy()}

    # weight data uncertainty
    sigma = np.concatenate((np.ones(Sxx.shape)*np.max(np.abs(Sxx))/np.max(np.abs(Sxy)), np.ones(Sxy.shape)))

    # start fitting with different models
    for name, use, model, verify, post_param in models:
        if not use:
            continue
        try:
            p, cov = curve_fit(model, BsSs, Ss, ps[name], sigma = sigma, bounds = (0, np.inf), maxfev = 5000)
            ps[name] = p.copy()

            # get fitted parameters and their errors, and post process parameter for each model
            uh, nh, ue, ne, sxx0 = p
            duh, dnh, due, dne, dsxx0 = np.sqrt(cov.diagonal())
            uh, nh, ue, ne, sxx0_sub, duh, dnh, due, dne, dsxx0_sub = post_param(Sxx, uh, nh, ue, ne, sxx0, duh, dnh, due, dne, dsxx0)

            # rescale back the parameters to SI unit
            nh *= S_scale/q
            ne *= S_scale/q
            dnh *= S_scale/q
            dne *= S_scale/q
            sxx0 *= S_scale
            dsxx0 *= S_scale

            # calculate residual of fitting
            Ss_diff = (verify(BsSs, *p) - Ss) / sigma 
            Ss_relative_diff = Ss_diff /np.max(np.abs(Sxy))
            l = int(len(Ss_diff) // 2)
            resxy = np.sqrt(np.mean((Ss_diff[0*l:1*l])**2))
            resxx = np.sqrt(np.mean((Ss_diff[1*l:2*l])**2))
            relresxy = np.sqrt(np.mean((Ss_relative_diff[0*l:1*l])**2))
            relresxx = np.sqrt(np.mean((Ss_relative_diff[1*l:2*l])**2))

            # recode data to file
            params_file = f'{resu_dir}model{name}_params.txt'
            with open(params_file, 'w' if i == 0 else 'a') as f:
                if i == 0:
                    params_files.append(params_file)
                    f.writelines(['T(K)\t',
                                    'uh(m^2/Vs)\t', 'duh(m^2/Vs)\t', 'nh(m^-3)\t', 'dnh(m^-3)\t',
                                    'ue(m^2/Vs)\t', 'due(m^2/Vs)\t', 'ne(m^-3)\t', 'dne(m^-3)\t',
                                    'sxx0(1/Om)\t', 'dsxx0(1/Om)\t', 'resxx\t', 'resxy\t', 'relresxx\t', 'relresxy\n'])
                f.writelines(['%.2f\t%.5e\t%.5e\t%.5e\t%.5e\t%.5e\t%.5e\t%.5e\t%.5e\t%.5e\t%.5e\t%.5f\t%.5f\t%.5f\t%.5f\n'
                                %(T, uh, duh, nh, dnh, ue, due, ne, dne, sxx0, dsxx0, resxx, resxy, relresxx, relresxy)])

            # plot fitting against data
            plot_prediction(f'{resu_dir}model{name}_predict_{T}K.png', T, verify, BsSs, p, S_scale)
            print(f'\n Sucessfully optimized fitting parameters for model {name}:')
        except:
            print(f'Unable to optimize model {name}.')
            continue

    print('--------------------------------------')

plot_sigma(Temperatures, Bxxs, Sxxs, Bxys, Sxys, resu_dir, S_scale)

# plot fitting parameters, residual for each model at different temperatures
plot_residual(f'{resu_dir}model_residual.png', params_files)
plot_relative_residual(f'{resu_dir}model_relative_residual.png', params_files)
plot_fitting_params(f'{resu_dir}_params.png', params_files, False)