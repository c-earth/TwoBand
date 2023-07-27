import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
cmap = get_cmap('tab20b')
colors = cmap.colors
colors = { 0: colors[0: 4],
           1: colors[4: 8],
           2: colors[12:16]}

q = 1.602E-19

def read_file(filename, Hmin = 0, Hmax = -1):
    B = []
    S = []
    with open(filename, 'r') as f:
        for line in f.readlines()[1:]:
            line_data =  line.split(',') 
            B.append(float(line_data[1]))
            S.append(float(line_data[2]))

    B = np.array(B)
    S = np.array(S)

    idxs = np.argsort(B)
    B = B[idxs]
    S = S[idxs]
    Lmin = int(Hmin/max(B) * len(B))
    if Hmax > Hmin:
        
        Lmax = int(Hmax/max(B) * len(B))

    else:
        Lmax = len(B)

    B = B[Lmin: Lmax]
    S = S[Lmin: Lmax]
    return B, S

def read_params(filename):
    with open(filename, 'r') as f:
        params = []
        for line in f.readlines()[1:]:
            params.append([float(x) for x in line.split('\t')])
    return np.array(params)

def plot_prediction(savename, T, model, BsSs, p, S_scale):
    Ss_pred = model(BsSs, *p)
    l = int(len(BsSs)/4)
    Bxx = BsSs[0*l:1*l]
    Bxy = BsSs[1*l:2*l]
    Sxx_true = BsSs[2*l:3*l] * S_scale
    Sxy_true = BsSs[3*l:4*l] * S_scale
    Sxx_pred = Ss_pred[0*l:1*l] * S_scale
    Sxy_pred = Ss_pred[1*l:2*l] * S_scale
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,8))
    
    ax1.plot(Bxx, Sxx_true, linestyle='', marker='.', markersize=10, linewidth=2, color=colors[0][0], label=f'expected $Sxx$, {T}K')
    ax1.plot(Bxx, Sxx_pred, linestyle='--', marker='', markersize=14, linewidth=2, color='r', label=f'predicted $Sxx$, {T}K')
    
    ax2.plot(Bxy, Sxy_true, linestyle='', marker='.', markersize=10, linewidth=2, color=colors[0][0], label=f'expected $Sxy$, {T}K')
    ax2.plot(Bxy, Sxy_pred, linestyle='--', marker='', markersize=14, linewidth=2, color='r', label=f'predicted $Sxy$, {T}K')
    
    ax1.set_xlabel(r'$B$ $(T)$', fontsize=20)
    ax1.set_ylabel(r'$\sigma_{xx}$ [1/$\Omega$m]', fontsize=20)
    ax1.legend(fontsize=20,loc='best')
    ax1.tick_params(axis='both', which='both', direction='in', labelsize=20, width=3, length=10)
    ax1.xaxis.get_offset_text().set_size(20)
    ax1.yaxis.get_offset_text().set_size(20)
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    ax2.set_xlabel(r'$B$ $(T)$', fontsize=20)
    ax2.set_ylabel(r'$\sigma_{xy}$ [1/$\Omega$m]', fontsize=20)
    ax2.legend(fontsize=20,loc='best')
    ax2.tick_params(axis='both', which='both', direction='in', labelsize=20, width=3, length=10)
    ax2.xaxis.get_offset_text().set_size(20)
    ax2.yaxis.get_offset_text().set_size(20)
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    f.tight_layout()
    f.savefig(savename, dpi=300)
    plt.close()

def plot_residual(savename, params_files):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,8))
    for params_file in params_files:
        params = read_params(params_file)
        name = params_file.split('/')[-1][:-11]
        ax1.plot(params[:, 0], params[:, -4], linestyle='-', marker='o', markersize=12, linewidth=3, label=name, color = colors[0][0])
        ax2.plot(params[:, 0], params[:, -3], linestyle='-', marker='o', markersize=12, linewidth=3, label=name, color = colors[0][0])

    ax1.set_xlabel(r'$T$ $(K)$', fontsize=20)
    ax1.set_ylabel(r'$\Delta\sigma_{xx}(RMSE)$', fontsize=20)
    # ax1.legend(fontsize=20, loc='best')
    ax1.tick_params(axis='both', which='both', direction='in', labelsize=20, width=3, length=10)
    ax1.xaxis.get_offset_text().set_size(20)
    ax1.yaxis.get_offset_text().set_size(20)

    ax2.set_xlabel(r'$T$ $(K)$', fontsize=20)
    ax2.set_ylabel(r'$\Delta\sigma_{xy}(RMSE)$', fontsize=20)
    # ax2.legend(fontsize=20, loc='best')
    ax2.tick_params(axis='both', which='both', direction='in', labelsize=20, width=3, length=10)
    ax2.xaxis.get_offset_text().set_size(20)
    ax2.yaxis.get_offset_text().set_size(20)

    f.tight_layout()
    f.savefig(savename, dpi=300)
    plt.close()

def plot_fitting_params(savename, params_files, errbar):
    for params_file in params_files:
        params = read_params(params_file)
        name = params_file.split('/')[-1][:-11]

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,8))
            
        if errbar:
            ax1.errorbar(params[:, 0], params[:, 7], params[:, 8],linestyle='--',marker='o',markersize=14,linewidth=3,label='$n_e$')
            ax1.errorbar(params[:, 0], params[:, 3], params[:, 4],linestyle='--',marker='o',markersize=14,linewidth=3,label='$n_h$')
            ax2.errorbar(params[:, 0], params[:, 5], params[:, 6],linestyle='--',marker='s',markersize=14,linewidth=3,label='$\mu_e$')
            ax2.errorbar(params[:, 0], params[:, 1], params[:, 2],linestyle='--',marker='s',markersize=14,linewidth=3,label='$\mu_h$')
        else:
            ax1.plot(params[:, 0], params[:, 7],linestyle='--',marker='o',markersize=14,linewidth=3,label='$n_e$', color = colors[1][0])
            ax1.plot(params[:, 0], params[:, 3],linestyle='--',marker='o',markersize=14,linewidth=3,label='$n_h$', color = colors[2][0])
            ax2.plot(params[:, 0], params[:, 5],linestyle='--',marker='s',markersize=14,linewidth=3,label='$\mu_e$', color = colors[1][0])
            ax2.plot(params[:, 0], params[:, 1],linestyle='--',marker='s',markersize=14,linewidth=3,label='$\mu_h$', color = colors[2][0])

        ax1.set_xlabel(r'$T$ $(K)$', fontsize=20)
        ax1.set_ylabel(r'$n$ $(m^{-3})$', fontsize=20)
        ax1.set_yscale('log')
        ax1.legend(fontsize=20, loc='center left')
        ax1.tick_params(axis='both', which='both', direction='in', labelsize=20, width=3, length=10)
        ax1.xaxis.get_offset_text().set_size(20)
        ax1.yaxis.get_offset_text().set_size(20)

        ax2.set_xlabel(r'$T$ $(K)$', fontsize=20)
        ax2.set_ylabel(r'$\mu$ $(m^2/Vs)$', fontsize=20)
        ax2.set_yscale('log')
        ax2.legend(fontsize=20, loc='center right')
        ax2.tick_params(axis='both', which='both', direction='in', labelsize=20, width=3, length=10)
        ax2.xaxis.get_offset_text().set_size(20)
        ax2.yaxis.get_offset_text().set_size(20)
        
        f.tight_layout()
        f.savefig(savename[:-11] + name + savename[-11:], dpi=300)
        plt.close()

def plot_sigma(Ts, Bxxs, Sxxs, Bxys, Sxys, resu_dir, S_scale):
    colors = mpl.colormaps['gnuplot'](np.linspace(0.1, 0.9, int(np.max(Ts)) + 1))
    cmap = mpl.colors.LinearSegmentedColormap.from_list('custom_gnuplot', colors)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,8))
    for T, Bxx, Sxx, Bxy, Sxy in zip(Ts, Bxxs, Sxxs, Bxys, Sxys):
        ax1.plot(Bxx, Sxx * S_scale, '-', color = colors[int(T)], linewidth = 3)
        ax2.plot(Bxy, Sxy * S_scale, '-', color = colors[int(T)], linewidth = 3)

    ax1.set_xlabel(r'$B$ [T]', fontsize = 20)
    ax1.set_ylabel(r'$\sigma_{xx}$ [$1/\Omega$m]', fontsize = 20)
    ax1.tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 20)
    ax1.set_xlim((np.min(Bxx), np.max(Bxx)))
    ax1.yaxis.get_offset_text().set_size(20)

    ax2.set_xlabel(r'$B$ [T]', fontsize = 20)
    ax2.set_ylabel(r'$\sigma_{xy}$ [$1/\Omega$m]', fontsize = 20)
    ax2.tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 20)
    ax2.set_xlim((np.min(Bxy), np.max(Bxy)))
    ax2.yaxis.get_offset_text().set_size(20)

    cb = f.colorbar(plt.cm.ScalarMappable(cmap = cmap, norm = plt.Normalize(vmin = 0, vmax = np.max(Ts))))
    cb.ax.set_title(r'$T$ (K)', fontsize = 20)
    cb.ax.tick_params(length = 5, width = 1.5, labelsize = 20)

    f.tight_layout()
    f.savefig(os.path.join(resu_dir, 'sigma_vs_B.png'))
    plt.close()

def plot_relative_residual(savename, params_files):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,8))
    for params_file in params_files:
        params = read_params(params_file)
        name = params_file.split('/')[-1][:-11]
        ax1.plot(params[:, 0], params[:, -2], linestyle='-', marker='o', markersize=12, linewidth=3, label=name, color = colors[0][0])
        ax2.plot(params[:, 0], params[:, -1], linestyle='-', marker='o', markersize=12, linewidth=3, label=name, color = colors[0][0])

    ax1.set_xlabel(r'$T$ $(K)$', fontsize=20)
    ax1.set_ylabel(r'$\Delta\sigma_{xx}(RMSRE)$', fontsize=20)
    # ax1.legend(fontsize=20, loc='best')
    ax1.tick_params(axis='both', which='both', direction='in', labelsize=20, width=3, length=10)
    ax1.xaxis.get_offset_text().set_size(20)
    ax1.yaxis.get_offset_text().set_size(20)

    ax2.set_xlabel(r'$T$ $(K)$', fontsize=20)
    ax2.set_ylabel(r'$\Delta\sigma_{xy}(RMSRE)$', fontsize=20)
    # ax2.legend(fontsize=20, loc='best')
    ax2.tick_params(axis='both', which='both', direction='in', labelsize=20, width=3, length=10)
    ax2.xaxis.get_offset_text().set_size(20)
    ax2.yaxis.get_offset_text().set_size(20)

    f.tight_layout()
    f.savefig(savename, dpi=300)
    plt.close()