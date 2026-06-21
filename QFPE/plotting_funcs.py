'''
Helper functions to make figures in paper
'''
import Hamiltonians as Hams

import functools
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

plt.rcParams["mathtext.fontset"]="cm"
plt.rcParams["font.family"] = "STIXGeneral"


func_to_color = {
        Hams.GQFPE:"bo",
        Hams.GQFPE_S:"rs",
        Hams.GQFPE_SH:"k^",
        Hams.CL:"gx",
        }

def plot_expectations(times, all_expectations, legend=True, loc="lower right", colors = ["bo", "rs", "k^", "gx"],
        fillstyles=["none", "none", "none", "none"], save_name="test"):
    '''
    Plots trace and expectation values of 
    :math:`\hat x, \hat p, \hat x^2, \hat p^2, \hat x \hat p + \hat p\hat x
    '''
    

    y_labels = [
            r"$\Delta$Norm",
            r"$\langle \hat q \rangle$",
            r"$\langle \hat p \rangle$",
            r"$\langle \hat q^2 \rangle - \langle \hat q \rangle^2 $",
            r"$\langle \hat p^2\rangle -\langle \hat p \rangle^2$",
            r"$\langle \hat q \hat p + \hat p \hat q \rangle$",
            ]
    #(12,8)
    fig, ax = plt.subplots(3, 2, figsize=(10,7), sharex=True, constrained_layout=True)
    for i in range(len(all_expectations)):
        if i == 0:
            labels = y_labels
        else:
            labels = [""] * 6
        expectations = all_expectations[i]
        
        #expectations[0] = [expectations[0][i] - 1 for i in range(len(expectations[0]))] #commented out for diff
        num_points = len(expectations[0])
        x_var = [expectations[3][j] - expectations[1][j]**2 for j in range(num_points)]
        p_var = [expectations[4][j] - expectations[2][j]**2 for j in range(num_points)]
        expects_to_coord = {
            (0,1):expectations[5],
            (1,0):expectations[1],
            (1,1):expectations[2],
            (2,0):x_var,
            (2,1):p_var
            }
        for subax in expects_to_coord:
            max_val = max([abs(expects_to_coord[subax][i]) for i in range(len(expects_to_coord[subax]))])
            power = 0
            while max_val * 10**(-power) < 1:
                power -= 1
            for j in range(len(expects_to_coord[subax])):
                expects_to_coord[subax][j] = expects_to_coord[subax][j] * 10 ** (-power)
            print(max_val*10**(-power))
            sci_formatter_new = functools.partial(sci_formatter, power=-power)
            ax[subax].yaxis.set_major_formatter(ticker.FuncFormatter(sci_formatter_new))
            #ax[subax].set_ylim(-.1,.1) 

        pow_scaling = -7
        expectations[0] = [expectations[0][i] * 10**-pow_scaling for i in range(len(expectations[0]))]
        ax[0][0].plot(times, expectations[0], colors[i], markerfacecolor=colors[i][0], markerfacecoloralt="white",markeredgecolor=colors[i][0], fillstyle=fillstyles[i], label=labels[0], ms=4)
    #produces correct scaling for remainder of ax besides trace
        ax[0][1].plot(times, expectations[5], colors[i], markerfacecolor=colors[i][0], markerfacecoloralt="white",markeredgecolor=colors[i][0], fillstyle=fillstyles[i], label=labels[5], markersize=4)
        ax[1][0].plot(times, expectations[1], colors[i], markerfacecolor=colors[i][0], markerfacecoloralt="white",markeredgecolor=colors[i][0], fillstyle=fillstyles[i], label=labels[1], markersize=4)
        ax[1][1].plot(times, expectations[2], colors[i], markerfacecolor=colors[i][0], markerfacecoloralt="white",markeredgecolor=colors[i][0], fillstyle=fillstyles[i], label=labels[2], markersize=4)
         
       # x_var = [expectations[3][j] - expectations[1][j]**2 for j in range(num_points)]
        #x_var = expectations[3]
        #print(x_var[0], "is <x^2>(0)")
        ax[2][0].plot(times, x_var, colors[i], markerfacecolor=colors[i][0], markerfacecoloralt="white", fillstyle=fillstyles[i], label=labels[3], ms=4)
       # p_var = [expectations[4][j] - expectations[2][j]**2 for j in range(num_points)]
        #p_var = expectations[4]
        #print(p_var[0], "is <p^2>(0)")
        ax[2][1].plot(times, p_var, colors[i], markerfacecolor=colors[i][0], markerfacecoloralt="white", fillstyle=fillstyles[i], label=labels[4], ms=4)
        print(expectations[5][0], "is xp_px_0")
         
        
    
    sci_formatter_new = functools.partial(sci_formatter,power=-pow_scaling)
    ax[0,0].yaxis.set_major_formatter(ticker.FuncFormatter(sci_formatter_new))
    ax[0,0].set_ylim(-1, 1)
    



    #ax[0][0].ticklabel_format(axis='y', style='plain', useOffset=False)
    for i in range(len(ax[0])):
        for j in range(len(ax)):
            if j == 0 and i == 1:
                #legend = ax[j][i].legend(fontsize=22, handlelength=0, edgecolor=None, facecolor=None, scatterpoints=0, handletextpad=0, loc="upper right")
                legend = ax[j][i].legend(fontsize=22, handlelength=0, edgecolor=None, facecolor=None, scatterpoints=0, handletextpad=0, loc="lower right")
            elif j == 2 and i == 0:
                #legend = ax[j][i].legend(fontsize=22, handlelength=0, edgecolor=None, facecolor=None, scatterpoints=0, handletextpad=0, loc="center right")
                #legend = ax[j][i].legend(fontsize=22, handlelength=0, edgecolor=None, facecolor=None, scatterpoints=0, handletextpad=0, loc="lower right")
                legend = ax[j][i].legend(fontsize=22, handlelength=0, edgecolor=None, facecolor=None, scatterpoints=0, handletextpad=0, loc="upper right")
            elif j == 1:
                legend = ax[j][i].legend(fontsize=20, handlelength=0, edgecolor=None, facecolor=None, scatterpoints=0, handletextpad=0, loc="upper right")
            elif j == 2 and i == 1:
                legend = ax[j][i].legend(fontsize=22, handlelength=0, edgecolor=None, facecolor=None, scatterpoints=0, handletextpad=0, loc="lower right")
            else:
                legend = ax[j][i].legend(fontsize=22, handlelength=0, edgecolor=None, facecolor=None, scatterpoints=0, handletextpad=0)
            for item in legend.legendHandles:
                item.set_visible(False)
            ax[j][i].tick_params(axis="both",labelsize=22)
            ax[j][i].locator_params(axis="both", nbins=5)
        if j == 2:
            ax[j][i].set_xlabel(r"$t$", fontsize=23)
    #ax[0][0].set_ylim(1-10**-7, 1+10**-7)
    os.makedirs("Figures/expectations", exist_ok=True)
    fig.savefig(f"Figures/expectations/{save_name}", dpi=300, bbox_inches='tight')
    
    return fig, ax

def plot_uncertainty(times, all_expectations, legend=True, loc="lower right", colors = ["bo", "rs", "k^", "gx"],
        fillstyles=["none", "none", "none", "none"], save_name="test"):
    '''
    Plots expectation values of 
    :math:`[\hat x, \hat p], {\hat x, \hat p}, and the generalized uncertainty principle
    '''
    

    y_labels = [
            r"$\langle [\hat x, \hat p] \rangle$",
            r"$\langle {\hat x, \hat p} \rangle$",
            r"$\Delta_{xx}\Delta_{pp}\rangle$", 
            r"$\sigma_{xx}\sigma_{pp}\rangle$",
            r"$\text{Tr}(\rho^2)$",
            # r"$\langle \hat q \rangle$",
           # r"$\langle \hat p \rangle$",
           # r"$\langle \hat q^2 \rangle - \langle \hat q \rangle^2 $",
           # r"$\langle \hat p^2\rangle -\langle \hat p \rangle^2$",
           # r"$\langle \hat q \hat p + \hat p \hat q \rangle$",
            ]
    #(12,8)
    fig, ax = plt.subplots(3, 2, figsize=(7,11), sharex=True, constrained_layout=True)
    for i in range(len(all_expectations)):
        if i == 0:
            labels = y_labels
        else:
            labels = [""] * 4
        expectations = all_expectations[i]
        uncertainty_plot = []
        uncertainty_plot.append([expectations[6][i].imag for i in range(len(expectations[0]))])
        uncertainty_plot.append(expectations[5])
        uncertainty_plot.append([(expectations[6][i]/2j)**2 + ((expectations[5][i]/2) - expectations[1][i]*expectations[2][i])**2 - 0.25 for i in range(len(expectations[0]))])
        print(expectations[6][0]/2j**2)
        print((expectations[5][0]/2 - expectations[1][0]*expectations[2][0])**2)
        uncertainty_plot.append([(expectations[3][i] - expectations[1][i]**2)*(expectations[4][i]-expectations[2][i]**2) for i in range(len(expectations[0]))])
        uncertainty_plot.append(expectations[8])
        expectations = uncertainty_plot
        #expectations[0] = [expectations[0][i] - 1 for i in range(len(expectations[0]))]
        #pow_scaling = -7
        #expectations[0] = [expectations[0][i] * 10**-pow_scaling for i in range(len(expectations[0]))]
        ax[0][0].plot(times, expectations[0], colors[i], markerfacecolor=colors[i][0], markerfacecoloralt="white",markeredgecolor=colors[i][0], fillstyle=fillstyles[i], label=labels[0], ms=4)
        ax[0][1].plot(times, expectations[1], colors[i], markerfacecolor=colors[i][0], markerfacecoloralt="white",markeredgecolor=colors[i][0], fillstyle=fillstyles[i], label=labels[1], markersize=4)
        ax[1][0].plot(times, expectations[2], colors[i], markerfacecolor=colors[i][0], markerfacecoloralt="white",markeredgecolor=colors[i][0], fillstyle=fillstyles[i], label=labels[2], markersize=4)
        ax[1][1].plot(times, expectations[3], colors[i], markerfacecolor=colors[i][0], markerfacecoloralt="white",markeredgecolor=colors[i][0], fillstyle=fillstyles[i], label=labels[3], markersize=4)
        ax[2][1].plot(times, expectations[4], colors[i], markerfacecolor=colors[i][0], markerfacecoloralt="white",markeredgecolor=colors[i][0], fillstyle=fillstyles[i], label=labels[4], markersize=4)
        #num_points = len(expectations[0])
        #x_var = [expectations[3][j] - expectations[1][j]**2 for j in range(num_points)]
        #x_var = expectations[3]
        #print(x_var[0], "is <x^2>(0)")
        #ax[2][0].plot(times, x_var, colors[i], markerfacecolor=colors[i][0], markerfacecoloralt="white", fillstyle=fillstyles[i], label=labels[3], ms=4)
        #p_var = [expectations[4][j] - expectations[2][j]**2 for j in range(num_points)]
        #p_var = expectations[4]
        #print(p_var[0], "is <p^2>(0)")
        #ax[2][1].plot(times, p_var, colors[i], markerfacecolor=colors[i][0], markerfacecoloralt="white", fillstyle=fillstyles[i], label=labels[4], ms=4)
        #print(expectations[5][0], "is xp_px_0")
    
    
    #sci_formatter_new = functools.partial(sci_formatter,power=-pow_scaling)
    #ax[0,0].yaxis.set_major_formatter(ticker.FuncFormatter(sci_formatter_new))
    #ax[0,0].set_ylim(-1, 1)
    for i in range(len(ax[0])):
        for j in range(len(ax)):
            if j == 0 and i == 1:
                legend = ax[j][i].legend(fontsize=22, handlelength=0, edgecolor=None, facecolor=None, scatterpoints=0, handletextpad=0, loc="upper right")
                #legend = ax[j][i].legend(fontsize=22, handlelength=0, edgecolor=None, facecolor=None, scatterpoints=0, handletextpad=0, loc="lower right")
            elif j == 2 and i == 0:
                legend = ax[j][i].legend(fontsize=22, handlelength=0, edgecolor=None, facecolor=None, scatterpoints=0, handletextpad=0, loc="center right")
            else:
                legend = ax[j][i].legend(fontsize=22, handlelength=0, edgecolor=None, facecolor=None, scatterpoints=0, handletextpad=0)
            for item in legend.legendHandles:
                item.set_visible(False)
            ax[j][i].tick_params(axis="both",labelsize=22)
            ax[j][i].locator_params(axis="both", nbins=5)
        if j == 2:
            ax[j][i].set_xlabel(r"$t$", fontsize=23)
    #ax[0][0].set_ylim(1-10**-7, 1+10**-7)
    os.makedirs("Figures/uncertainty", exist_ok=True)
    fig.savefig(f"Figures/uncertainty/{save_name}", dpi=300, bbox_inches='tight')
    
    return fig, ax

def plot_rho_coord(times, rho_coords, x_vals = np.linspace(-10,10, 512), y_vals=np.linspace(-10,10,512), real=True, save_name="test"):
    fs=23
    num_rows = len(rho_coords)
    num_cols = len(times)
    func = np.real if real else np.imag
    #figsize=(12,10) for I
    #figsize=(12,8) for II is OK
    fig, ax = plt.subplots(num_rows, num_cols, figsize=(12, 8), constrained_layout=True, sharex=True, sharey=True)
    #plt.subplots_adjust(wspace=0.04, hspace=0.04)
    #x_vals = np.linspace(-10, 10, 512)
    #y_vals = x_vals
    func_names = [
            #"GQFPE",
            "GQFPE-S",
            "GQFPE-SH",
            "CL-QFPE"
            ]


    for row in range(num_rows):
        for col in range(num_cols):
            if row == 0:
                ax[row][col].text(.5, 1.05, rf"$t={times[col]:.2f}$", ha="center", va="bottom", transform=ax[row][col].transAxes, fontsize=fs)
            if row == num_rows - 1:
                ax[row][col].set_xlim(-8, 8)
                ax[row][col].set_xlabel(r"$q'$", fontsize=fs)
            if col == 0:
                ax[row][col].set_ylim(-8,8)
                ax[row][col].set_ylabel(r"$q''$", fontsize=fs)
                ax[row][col].text(-.15,.75, f"{func_names[row]}" , va="center", ha="right", transform=ax[row][col].transAxes, fontsize=fs)
            two_d_plot = ax[row][col].contourf(x_vals, y_vals, func(rho_coords[row][col]), cmap="seismic", levels=np.linspace(-.5,.5,101))
            if row == 0 and col == 0:
                #cax = fig.add_axes([0.96, 0.15, 0.02, 0.7])
                cb = fig.colorbar(two_d_plot, ax=ax, location="right", shrink=0.6, pad=0.02)
                cb.ax.tick_params(labelsize=fs)
            ax[row,col].tick_params(axis="both",labelsize=fs)
            ax[row,col].locator_params(axis="both", nbins=6)
            ax[row,col].set_aspect('equal')
    
    os.makedirs("Figures/rho_coord", exist_ok=True)
    suffix = "real" if real else "imag"
    fig.savefig(f"Figures/rho_coord/{save_name}_{suffix}", dpi=300, bbox_inches="tight")
    
    return fig, ax



def plot_expectations_Ehrenfest(times, times_Ehrenfest, all_expectations, Ehrenfest_expectations, func= Hams.GQFPE, legend=True, loc="lower right", 
        fillstyles=["none", "none", "none", "none"], save_name="test"):
    color = func_to_color[func]
    Ehrenfest_color = "C09"
    
    y_labels = [
            r"$\Delta$Norm",
            r"$\langle \hat{q}\hat{p} + \hat{p}\hat{q}\rangle$",
            r"$\langle \hat{q}\rangle$",
            r"$\langle \hat{p}\rangle$",
            r"$\langle \hat{q}^2\rangle -\langle \hat{q} \rangle^2$",
            r"$\langle \hat{p}^2\rangle - \langle \hat{p} \rangle^2$",
            ]

    fig, ax = plt.subplots(3,2, figsize=(10,7), constrained_layout=True, sharex=True)
    all_expectations[0] = [all_expectations[0][i] - 1 for i in range(len(all_expectations[0]))]
    pow_scaling = -7
    Ehrenfest_expectations[0] = (Ehrenfest_expectations[0] - 1)*10**(-pow_scaling)
    all_expectations[0] = [all_expectations[0][i]*10**(-pow_scaling) for i in range(len(all_expectations[0]))]
    print(Ehrenfest_expectations[0])
    print(all_expectations[0][0])
    #ax[0,0].ticklabel_format(axis='y', style="sci", scilimits=(0,0))
    ax[0][0].plot(times, all_expectations[0], color, ms=4,  label=y_labels[0], fillstyle='none')
    ax[0][1].plot(times, all_expectations[5], color, ms=4,  label=y_labels[1], fillstyle='none')
    ax[1][0].plot(times, all_expectations[1], color, ms=4,  label=y_labels[2], fillstyle='none')
    ax[1][1].plot(times, all_expectations[2], color, ms=4,  label=y_labels[3], fillstyle='none')
         
    num_points = len(all_expectations[0])
    x_var = [all_expectations[3][j] - all_expectations[1][j]**2 for j in range(num_points)]
        #x_var = all_expectations[3]
        #print(x_var[0], "is <x^2>(0)")
    ax[2][0].plot(times, x_var, color, ms=4, label=y_labels[4], fillstyle='none')
    p_var = [all_expectations[4][j] - all_expectations[2][j]**2 for j in range(num_points)]
        #p_var = all_expectations[4]
        #print(p_var[0], "is <p^2>(0)")
    ax[2][1].plot(times, p_var, color, ms=4, label=y_labels[5], fillstyle='none')

    
    ax[0][0].plot(times_Ehrenfest, Ehrenfest_expectations[0], Ehrenfest_color)
    ax[0][1].plot(times_Ehrenfest, Ehrenfest_expectations[5], Ehrenfest_color)
    ax[1][0].plot(times_Ehrenfest, Ehrenfest_expectations[1], Ehrenfest_color)
    ax[1][1].plot(times_Ehrenfest, Ehrenfest_expectations[2], Ehrenfest_color)
         
    num_points = len(Ehrenfest_expectations[0])
    x_var = [Ehrenfest_expectations[3][j] - Ehrenfest_expectations[1][j]**2 for j in range(num_points)]
        #x_var = Ehrenfest_expectations[3]
        #print(x_var[0], "is <x^2>(0)")
    ax[2][0].plot(times_Ehrenfest, x_var, Ehrenfest_color)
    p_var = [Ehrenfest_expectations[4][j] - Ehrenfest_expectations[2][j]**2 for j in range(num_points)]
        #p_var = Ehrenfest_expectations[4]
        #print(p_var[0], "is <p^2>(0)")
    ax[2][1].plot(times_Ehrenfest, p_var, Ehrenfest_color)

    for i in range(6):
        row_num = i//2
        col_num = i % 2
       # ax[row_num][col_num].plot(times, all_expectations[i], color, label=y_labels[i], ms=4, fillstyle='none')
        
       # ax[row_num][col_num].plot(times_Ehrenfest, Ehrenfest_expectations[i], Ehrenfest_color)
        if row_num == 2:
            ax[row_num][col_num].set_xlabel(r"$t$", fontsize=23)
        ax[row_num][col_num].tick_params(axis="both",labelsize=22)
        ax[row_num][col_num].locator_params(axis="both", nbins=6)
        legend = ax[row_num][col_num].legend(fontsize=22, handlelength = 0, handletextpad=0, loc="best")
        if row_num == 1 and col_num == 1: #and (func == Hams.CL or func == Hams.GQFPE_SH:
            legend = ax[row_num][col_num].legend(fontsize=22, handlelength = 0, handletextpad=0, loc="lower right")
        if (row_num == 2 and col_num == 0 or row_num == 2 and col_num == 1) and func != Hams.CL:
            legend = ax[row_num][col_num].legend(fontsize=22, handlelength = 0, handletextpad=0, loc="lower right")
        for item in legend.legendHandles:
            item.set_visible(False)
    sci_formatter_new = functools.partial(sci_formatter,power=-pow_scaling)
    ax[0,0].yaxis.set_major_formatter(ticker.FuncFormatter(sci_formatter_new))
    ax[0,0].tick_params(axis="y", labelsize=18) 
    ax[0][0].set_ylim(-1,1)
    dir_name = "Figures/expectations/analytical_comparison"
    os.makedirs(dir_name, exist_ok=True)
    fig.savefig(dir_name + "/" + save_name, dpi=300, bbox_inches="tight")
   
    return fig, ax

def plot_taylor_error(num_terms, arnoldi_error, taylor_error, linestyles=["bx", "r."], save_name="test"):
    fig, ax = plt.subplots(3, 1, figsize=(5,9), sharex=True, constrained_layout=True)
    for i in range(3):
        ax[i].set_ylabel(r"$\text{log}_{10}$(Error)", fontsize=23)
        ax[i].plot(num_terms, arnoldi_error[i], linestyles[0], label="Arnoldi Iteration", markersize=12)
        ax[i].plot(num_terms, taylor_error[i], linestyles[1], label="Taylor Series", markersize=12)
        ax[i].tick_params(axis="both", labelsize=15)
        ax[i].text(-.15, 0.95, f'({chr(97+i)})', transform=ax[i].transAxes, fontsize=15)
    ax[0].legend(fontsize=16)
    ax[2].set_xlabel("Number of Iterations", fontsize=23)
    ax[2].set_xticks([i * 25 for i in range(6)]) 
    os.makedirs("Figures/taylor_error", exist_ok=True)
    fig.savefig(f"Figures/taylor_error/{save_name}", dpi=300, bbox_inches="tight")

    return fig, ax

def sci_formatter(x, pos, power=7):
    if x == 0:
        return "0"
    return (f"{x:.1f}" + rf"$\times 10^{{{-power}}}$")
     
