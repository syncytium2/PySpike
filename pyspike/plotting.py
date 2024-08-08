import matplotlib.pyplot as plt
import pyspike as spk
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from decimal import Decimal
from .generate_surrogate import generate_surro
from .latency_correction import Spike_time_difference_matrix, sim_ann_latency_correction, latency_correction_extrapol, latency_correction_intrapol
from .generic import Multi_Profile_Matrix
from .spikes import f_all_trains

def plot_matrix(matrix, variable, variable_value=None, showing=0, ax=None, vmin=None, vmax=None):
    """
    Plots a matrix representing spike train data.

    :param matrix: The matrix to be plotted.
    :type matrix: numpy.ndarray
    :param variable: The type of data represented by the matrix:
                     1 for ISI-distance,
                     2 for SPIKE-distance,
                     3 for Spike-Synchro,
                     4 for SPIKE-Order,
                     5 for Spike train order
                     6 for Spike time difference.
    :type variable: int
    :param variable_value: The value associated with the variable. If None, it's not displayed.
    :type variable_value: float or None
    :param showing: Determines whether to print the matrix before plotting (0 or 1).
    :type showing: int, optional
    :param ax: Matplotlib axis object to plot on, if None, creates a new one.
    :type ax: Matplotlib axis, optional
    :param vmin: Minimum data value that corresponds to the colormap's minimum color.
    :type vmin: float or None, optional
    :param vmax: Maximum data value that corresponds to the colormap's maximum color.
    :type vmax: float or None, optional

    Example::

            import matplotlib.pyplot as plt
            isi_distance = pyspike.isi_distance(spike_trains)
            isi_distance_mat = pyspike.isi_distance_matrix(spike_trains)
            pyspike.plot_matrix(isi_distance_mat, variable=1, variable_value=isi_distance, showing=0)
            plt.show()
    """

    possible_variable = ['ISI-distance', 'SPIKE-distance', 'Spike-Synchro', 'SPIKE-Order', 'Spike train order', 'Spike time difference']
    
    num_trains = len(matrix)
    if showing == 1:
        if variable_value is not None:
            print(f"\n{possible_variable[variable-1]}: %.8f\n" % (variable_value))
        else:
            print(f"\n{possible_variable[variable-1]}\n")
        for i in range(num_trains):
            print("\n%i     " % (i+1), end = "")
            for j in range(num_trains):
                print("%.8f " % (matrix[i][j]), end = "")
        print("\n")
        return 0

    if ax is None:
        fig, ax = plt.subplots(figsize=(17, 10), dpi=80)
    else:
        fig = ax.figure

    if num_trains > 50:
        div = 10
    elif num_trains > 10:
        div = 5
    else:
        div = 1
    ticks = list(range(num_trains, 1, -div))
    if ticks[-1] != 1:
        ticks.append(1)
    str_ticks = [str(i) for i in ticks]
    ticks1 = []
    for i in range(len(ticks)):
        ticks1.append(ticks[i]-1)
    
    cax = plt.imshow(matrix, interpolation='none', vmin=vmin, vmax=vmax)
    if variable_value is not None:
        plt.title("%s Matrix (%s = %.8f)" % (possible_variable[variable-1], possible_variable[variable-1], variable_value), color='k', fontsize=24)
    else:
        plt.title("%s Matrix" % possible_variable[variable-1], color='k', fontsize=24)
    plt.xlabel('Spike Trains', color='k', fontsize=18)
    plt.ylabel('Spike Trains', color='k', fontsize=18)
    plt.yticks(ticks1)
    ax.set_yticklabels(str_ticks, fontsize=14)
    plt.xticks(ticks1)
    ax.set_xticklabels(str_ticks, fontsize=14)
    plt.jet()
    colorbar = plt.colorbar()

    return fig, ax, cax, colorbar

def plot_profile(x_prof, y_prof, variable, variable_value=None, showing=0):
    """
    Plots a profile representing spike train data.

    :param x_prof: The x-values of the profile.
    :type x_prof: list or numpy.ndarray
    :param y_prof: The y-values of the profile.
    :type y_prof: list or numpy.ndarray
    :param variable: The type of data represented by the profile:
                     1 for ISI-distance,
                     2 for SPIKE-distance,
                     3 for Spike-Synchro,
                     4 for SPIKE-Order,
                     5 for Spike train order.
    :type variable: int
    :param variable_value: The value associated with the variable. If None, it's not displayed.
    :type variable_value: float or None
    :param showing: Determines whether to print the profile before plotting (0 or 1).
    :type showing: int, optional

    Example::

            import matplotlib.pyplot as plt
            isi_distance = pyspike.isi_distance(spike_trains)
            isi_profile = pyspike.isi_profile(spike_trains)
            x, y = isi_profile.get_plottable_data()
            pyspike.plot_profile(x, y, variable=1, variable_value=isi_distance, showing=0)
            plt.show()
    """

    possible_variable = ['ISI-distance', 'SPIKE-distance', 'Spike-Synchro', 'SPIKE-Order', 'Spike train order']
    tmin = x_prof[0]
    tmax = x_prof[-1]

    if showing == 1:
        print(f"\n{possible_variable[variable-1]}: %.8f\n" % (variable_value))
        print("\n%s Profile:\n" %(possible_variable[variable-1]))
        print("x            y\n")
        for i in range(len(x_prof)):
            print("%.8f   %.8f\n" % (x_prof[i], y_prof[i]), end = "")
        print("\n")
        return 0

    plt.figure(figsize=(17, 10), dpi=80)
    plt.plot(x_prof, y_prof, '-k*')
    
    if variable == 4 or variable == 5:
        plt.axis([tmin-0.05*(tmax-tmin), tmax+0.05*(tmax-tmin), -1.05, 1.05])
        plt.plot((tmin, tmax), (0, 0), ':', color='k', linewidth=1)
        plt.plot((tmin, tmax), (1, 1), ':', color='k', linewidth=1)
        plt.plot((tmin, tmax), (-1, -1), ':', color='k', linewidth=1)
        plt.plot((tmin, tmin), (-1, 1), ':', color='k', linewidth=1)
        plt.plot((tmax, tmax), (-1, 1), ':', color='k', linewidth=1)
    else:
        plt.axis([tmin-0.05*(tmax-tmin), tmax+0.05*(tmax-tmin), -0.05, 1.05])
        plt.plot((tmin, tmax), (0, 0), ':', color='k', linewidth=1)
        plt.plot((tmin, tmax), (1, 1), ':', color='k', linewidth=1)
        plt.plot((tmin, tmin), (0, 1), ':', color='k', linewidth=1)
        plt.plot((tmax, tmax), (0, 1), ':', color='k', linewidth=1)
    plt.plot((tmin, tmax), (variable_value, variable_value), '--', color='k', linewidth=1)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    if variable_value is not None:
        plt.title("%s Profile (%s = %.8f)" % (possible_variable[variable-1], possible_variable[variable-1], variable_value), color='k', fontsize=24)
    else:
        plt.title("%s Profile" % possible_variable[variable-1], color='k', fontsize=24)
    
    plt.xlabel('Time', color='k', fontsize=18)
    plt.ylabel("%s" %(possible_variable[variable-1]), color='k', fontsize=18)

def plot_spike_trains(spikes, phi=None, showing=0, order_color=0, ax=None):
    """
    Plots spike trains in a raster plot.

    :param spikes: List of spike trains.
    :type spikes: List of :class:`pyspike.SpikeTrain`
    :param phi: Order of spike trains to be plotted, if None, default order is used.
    :type phi: list or None, optional
    :param showing: Determines whether to print the spike trains before plotting (0 or 1).
    :type showing: int, optional
    :param order_color: Determines whether to plot spike trains with color based on their order (0 or 1).
    :type order_color: int, optional
    :param ax: Matplotlib axis object to plot on, if None, creates a new one.
    :type ax: Matplotlib axis, optional

    Example::

            import matplotlib.pyplot as plt
            pyspike.plot_spike_trains(spike_trains, showing=0, order_color=1)
            plt.show()
    """
    tmin = spikes[0].t_start
    tmax = spikes[0].t_end
    num_trains = len(spikes)
    if showing == 1:
        if phi == None:
            for i in range(num_trains):
                print("\nSpike Train %3i:" % (i+1))
                for j in range(len(spikes[i])):
                    print("%i %.8f" % (j+1, spikes[i][j]))
            print("\n")
        else:
            for i in phi:
                print("\nSpike Train %3i:" % (i+1))
                for j in range(len(spikes[i])):
                    print("%i %.8f" % (j+1, spikes[i][j]))
            print("\n")
        return 0
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(17, 10), dpi=80)
    else:
        fig = ax.figure

    plotted_spikes = []
    colorbar = None
    cax = None

    if num_trains > 50:
        div = 10
    elif num_trains > 10:
        div = 5
    else:
        div = 1
        
    yticks = list(range(1, num_trains + 1, div))
    if yticks[-1] != num_trains:
        yticks.append(num_trains)
    ax.set_yticks(range(1, num_trains+1))
    ax.set_yticklabels([str(num_trains - i) for i in range(num_trains)])

    if order_color == 1:
        D = spk.spike_order_values(spikes)
        spike_sync_profile = spk.spike_sync_profile(spikes)
        C = spike_sync_profile.get_multi_plottable_data(spikes)[1]
        colors = [(0, 0, 0.5), (0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0.5, 0), (1, 0, 0), (0.5, 0, 0)]
        positions = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0]
        cm = LinearSegmentedColormap.from_list('custom_cmap', list(zip(positions, colors)), N=256)
        cax = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=-1, vmax=1))
        colorbar = plt.colorbar(cax, ax=ax)
        plotted_spikes.append(colorbar)

        order = f_all_trains(spikes)[0]

        if phi == None:
            indexed_liste1 = list(enumerate(order))
            zipped_lists = list(zip(indexed_liste1, C))
            sorted_zipped_lists = sorted(zipped_lists, key=lambda x: (x[0][1], x[0][0]))
            _, sorted_C = zip(*sorted_zipped_lists)
            plt.title("Rasterplot", color='k', fontsize=24)
        else:
            plt.title("Sorted Rasterplot", color='k', fontsize=24)
            zipped_lists = list(zip(order, C))
            sorted_zipped_lists = sorted(zipped_lists, key=lambda x: phi[x[0] - 1])
            _, sorted_C = zip(*sorted_zipped_lists)
            sorted_C = list(sorted_C)

    
    plt.xlabel('Time', color='k', fontsize=18)
    plt.ylabel('Spike Trains', color='k', fontsize=18)
    plt.axis([tmin-0.05*(tmax-tmin), tmax+0.05*(tmax-tmin), 0, num_trains+1])
    plt.xticks(fontsize=14)

    if phi==None:
        plt.yticks(yticks, fontsize=14)
    else:
        plt.yticks(np.arange(num_trains)+1, reversed([x+1 for x in phi]), fontsize=14)

    plotted_spikes.append(plt.plot((tmin, tmax), (0.5, 0.5), ':', color='k', linewidth=1))
    plotted_spikes.append(plt.plot((tmin, tmax), (num_trains+0.5, num_trains+0.5), ':', color='k', linewidth=1))
    plotted_spikes.append(plt.plot((tmin, tmin), (0.5, num_trains+0.5), ':', color='k', linewidth=1))
    plotted_spikes.append(plt.plot((tmax, tmax), (0.5, num_trains+0.5), ':', color='k', linewidth=1))

    N = 0
    if phi == None:
        for i in range(num_trains):
            for j in range(len(spikes[i])):
                if order_color == 1:
                    color = cm((D[i][j] + 1) / 2)
                    if sorted_C[N] == 0:
                        plotted_spikes.append(plt.plot((spikes[i][j], spikes[i][j]), (num_trains-i+0.5, num_trains-i-.5), '-', color='k', linewidth=1))
                    else:
                        plotted_spikes.append(plt.plot((spikes[i][j], spikes[i][j]), (num_trains-i+0.5, num_trains-i-.5), '-', color=color, linewidth=1+2*sorted_C[N]))
                else:
                    plotted_spikes.append(plt.plot((spikes[i][j], spikes[i][j]), (num_trains-i+0.5, num_trains-i-.5), '-', color='k', linewidth=1))
                N += 1
    else:
        for i in range(num_trains):
            for j in range(len(spikes[phi[i]])):
                if order_color == 1:
                    color = cm((D[phi[i]][j] + 1) / 2)
                    if sorted_C[N] == 0:
                        plotted_spikes.append(plt.plot((spikes[phi[i]][j], spikes[phi[i]][j]), (num_trains-i+0.5, num_trains-i-.5), '-', color='k', linewidth=1))
                    else:
                        plotted_spikes.append(plt.plot((spikes[phi[i]][j], spikes[phi[i]][j]), (num_trains-i+0.5, num_trains-i-.5), '-', color=color, linewidth=1+2*sorted_C[N]))
                else:
                    plotted_spikes.append(plt.plot((spikes[phi[i]][j], spikes[phi[i]][j]), (num_trains-i+0.5, num_trains-i-.5), '-', color='k', linewidth=1))
                N += 1
    return fig, ax, plotted_spikes, colorbar, cax

def spike_matching_plot(spikes):
    """
    Plots spike train matches against three specific spike trains: 
    the first, the middle, and the last spike train.

    :param spikes: List of spike trains.
    :type spikes: List of :class:`pyspike.SpikeTrain`

    Example::

            import matplotlib.pyplot as plt
            pyspike.spike_matching_plot(spike_trains)
            plt.show()
    """
    tmin = spikes[0].t_start
    tmax = spikes[0].t_end
    num_trains = len(spikes)
    yticklabel1, yticklabel2, yticklabel3, num_spikes = [], [], [], []
    STDM = Spike_time_difference_matrix(spikes)
    for i in range(num_trains):
        num_spikes.append(len(spikes[i]))
        yticklabel1.append(f'{num_trains-i}')
        yticklabel2.append(f'{num_trains-i}')
        yticklabel3.append(f'{num_trains-i}')
        if i == 0:
            yticklabel3.append('')
        if i == num_trains//2-1:
            yticklabel2.append('')
        if i == num_trains//2:
            yticklabel2.append('')
        if i == num_trains-2:
            yticklabel1.append('')

    # Plotting matches with spike train #1
    fig, ax, plotted_spikes, _, _ = plot_spike_trains(spikes, order_color=1)
    y_positions = np.arange(num_trains+1)

    for i, line in enumerate(plotted_spikes):
        if i > 4 + num_spikes[0]:
            line[0].set_ydata([y-1 for y in line[0].get_ydata().tolist()])
    
    ax.set_title("Matches with Spike Train #1", color='k', fontsize=24)
    plt.plot((tmin, tmax), (num_trains-1, num_trains-1), ':', color='k', linewidth=2)
    for i in range(1, num_trains):
        spike_train_order_profile = spk.spike_train_order_profile([spikes[0],spikes[i]])
        time, e_prof = spike_train_order_profile.get_multi_plottable_data([spikes[0],spikes[i]])
        j = 0
        while j < len(time)-1:
            if e_prof[j] == 1:
                plt.plot((time[j], time[j+1]), (num_trains-i-1, num_trains-i-1), '-', color='b', linewidth=2)
                j += 2
            elif e_prof[j] == -1:
                plt.plot((time[j], time[j+1]), (num_trains-i-1, num_trains-i-1), '-', color='r', linewidth=2)
                j += 2
            else:
                j += 1

    data = STDM[0]
    max_data_value = max(abs(min(data)), abs(max(data)))
    right_ax = _adjust_axes(fig, ax, plotted_spikes, max_data_value, y_positions, [-1, num_trains+1], yticklabel1)
    right_ax.plot((-max_data_value, max_data_value), (num_trains-1, num_trains-1), ':', color='k', linewidth=2)
    for i in range(num_trains):
        value = data[num_trains-i-1]
        color = 'blue' if value > 0 else 'red'
        right_ax.barh(y_positions[i], value, height=1, color=color, edgecolor='black')

    # Plotting matches with spike train #{num_trains//2}
    fig, ax, plotted_spikes, _, _ = plot_spike_trains(spikes, order_color=1)

    for i, line in enumerate(plotted_spikes):
        if i > 4 + sum(num_spikes[:num_trains//2]):
            line[0].set_ydata([y-2 for y in line[0].get_ydata().tolist()])
        elif i > 4 + sum(num_spikes[:num_trains//2-1]):
            line[0].set_ydata([y-1 for y in line[0].get_ydata().tolist()])

    ax.set_title(f"Matches with Spike Train #{num_trains//2}", color='k', fontsize=24)
    plt.plot((tmin, tmax), (num_trains//2-1, num_trains//2-1), ':', color='k', linewidth=2)
    plt.plot((tmin, tmax), (num_trains//2+1, num_trains//2+1), ':', color='k', linewidth=2)
    for i in range(num_trains//2-1):
        spike_train_order_profile = spk.spike_train_order_profile([spikes[num_trains//2-1],spikes[i]])
        time, e_prof = spike_train_order_profile.get_multi_plottable_data([spikes[num_trains//2-1],spikes[i]])
        j = 0
        while j < len(time)-1:
            if e_prof[j] == 1:
                plt.plot((time[j], time[j+1]), (num_trains-i, num_trains-i), '-', color='r', linewidth=2)
                j += 2
            elif e_prof[j] == -1:
                plt.plot((time[j], time[j+1]), (num_trains-i, num_trains-i), '-', color='b', linewidth=2)
                j += 2
            else:
                j += 1
    for i in range(num_trains//2, num_trains):
        spike_train_order_profile = spk.spike_train_order_profile([spikes[num_trains//2-1],spikes[i]])
        time, e_prof = spike_train_order_profile.get_multi_plottable_data([spikes[num_trains//2-1],spikes[i]])
        j = 0
        while j < len(time)-1:
            if e_prof[j] == 1:
                plt.plot((time[j], time[j+1]), (num_trains-i-2, num_trains-i-2), '-', color='b', linewidth=2)
                j += 2
            elif e_prof[j] == -1:
                plt.plot((time[j], time[j+1]), (num_trains-i-2, num_trains-i-2), '-', color='r', linewidth=2)
                j += 2
            else:
                j += 1

    data = STDM[num_trains//2-1]
    max_data_value = max(abs(min(data)), abs(max(data)))
    y_positions = []
    for i in range(num_trains+2):
        if not (i == num_trains//2 or i == num_trains//2+1):
            y_positions.append(i-1)
    right_ax = _adjust_axes(fig, ax, plotted_spikes, max_data_value, np.arange(-1, num_trains+1), [-2, num_trains+1], yticklabel2, matching=2)
    for i in range(num_trains):
        value = data[num_trains-i-1]
        if value != 0:
            color = 'blue' if value > 0 else 'red'
            if num_trains-i-1 < num_trains//2-1:
                right_ax.barh(y_positions[i], -value, height=1, color=color, edgecolor='black')
            else:
                right_ax.barh(y_positions[i], value, height=1, color=color, edgecolor='black')
    right_ax.plot((-max_data_value, max_data_value), (num_trains//2-1, num_trains//2-1), ':', color='k', linewidth=2)
    right_ax.plot((-max_data_value, max_data_value), (num_trains//2+1, num_trains//2+1), ':', color='k', linewidth=2)
    
    # Plotting matches with spike train #{num_trains}
    fig, ax, plotted_spikes, _, _ = plot_spike_trains(spikes, order_color=1)
    y_positions = np.arange(num_trains+1)
    
    for i, line in enumerate(plotted_spikes):
        if i >= len(plotted_spikes)-num_spikes[-1]:
            line[0].set_ydata([y-1 for y in line[0].get_ydata().tolist()])

    ax.set_title(f"Matches with Spike Train #{num_trains}", color='k', fontsize=24)
    plt.plot((tmin, tmax), (1, 1), ':', color='k', linewidth=2)
    for i in range(num_trains-1):
        spike_train_order_profile = spk.spike_train_order_profile([spikes[num_trains-1],spikes[i]])
        time, e_prof = spike_train_order_profile.get_multi_plottable_data([spikes[num_trains-1],spikes[i]])
        j = 0
        while j < len(time)-1:
            if e_prof[j] == 1:
                plt.plot((time[j], time[j+1]), (num_trains-i, num_trains-i), '-', color='r', linewidth=2)
                j += 2
            elif e_prof[j] == -1:
                plt.plot((time[j], time[j+1]), (num_trains-i, num_trains-i), '-', color='b', linewidth=2)
                j += 2
            else:
                j += 1

    data = STDM[-1]
    max_data_value = max(abs(min(data)), abs(max(data)))
    right_ax = _adjust_axes(fig, ax, plotted_spikes, max_data_value, y_positions, [-1, num_trains+1], yticklabel3)
    for i in range(num_trains):
        value = data[num_trains-i-1]
        if value != 0:
            color = 'blue' if value > 0 else 'red'
            right_ax.barh(y_positions[i]+1, -value, height=1, color=color, edgecolor='black')
    right_ax.plot((-max_data_value, max_data_value), (1, 1), ':', color='k', linewidth=2)

def _adjust_axes(fig, ax, plotted_spikes, max_data_value, y_positions, y_lim, yticklabel, matching=1):
    """
    Adjusts the main axes and adds a secondary axis to the right to display spike time differences.

    :param fig: The figure object to which axes are added.
    :type fig: matplotlib.figure.Figure
    :param ax: The main axes object.
    :type ax: matplotlib.axes.Axes
    :param plotted_spikes: The spike plot data.
    :type plotted_spikes: list
    :param max_data_value: The maximum absolute value of the data to be plotted on the secondary axis.
    :type max_data_value: float
    :param y_positions: The positions of the y-ticks.
    :type y_positions: list
    :param y_lim: The limits for the y-axis.
    :type y_lim: list
    :param yticklabel: The labels for the y-ticks.
    :type yticklabel: list
    :param matching: Determines the offset for the y-data adjustments. Default is 1.
    :type matching: int, optional
    :return: The secondary axis object.
    :rtype: matplotlib.axes.Axes
    """
    if matching == 1:
        num_trains = len(y_positions)-1
        y = -0.5
    else:
        num_trains = len(y_positions)-2
        y = -1.5
        
    plotted_spikes[0].remove()
    plotted_spikes[1][0].set_ydata([y, y])
    plotted_spikes[3][0].set_ydata([y, num_trains+0.5])
    plotted_spikes[4][0].set_ydata([y, num_trains+0.5])

    ax.set_yticks(y_positions)
    ax.set_ylim(y_lim[0], y_lim[1])
    ax.set_yticklabels(yticklabel)
    
    right_ax = fig.add_axes([0.8, 0.11, 0.15, 0.77])
    right_ax.set_xlim([-max_data_value, max_data_value])
    right_ax.invert_yaxis()
    right_ax.axvline(0, color='black', linewidth=1)
    right_ax.yaxis.tick_right()
    right_ax.set_yticks(y_positions)
    right_ax.set_ylim(y_lim[0], y_lim[1])
    right_ax.set_yticklabels(yticklabel)
    right_ax.set_title("Leaders   Followers", fontsize=18)
    right_ax.set_xlabel("<Spike Time Difference>", fontsize=18)
    return right_ax

def plot_surrogates(spike_trains, num_surros):
    """
    Plots the surrogate distribution of spike train order values and the optimal spike train order value.

    :param spike_trains: List of spike trains.
    :type spike_trains: List of :class:`pyspike.SpikeTrain`
    :param num_surros: Number of surrogates to generate.
    :type num_surros: int

    Example::

            import matplotlib.pyplot as plt
            pyspike.plot_surrogates(spike_trains, num_surros)
            plt.show()
    """
    sto_profs = Multi_Profile_Matrix(spike_trains, 3)
    values = generate_surro(sto_profs, num_surros)

    phi, _ = spk.optimal_spike_train_sorting(spike_trains)
    F_opt = spk.spike_train_order(spike_trains, indices=phi)

    num_interval = 100
    interval = 1/num_interval
    count = [0 for i in range(num_interval+1)]
    for i in range(len(values)):
        N = int(values[i]/interval)
        count[N] += 1
    mean_value = np.mean(values)
    std_dev = np.std(values)
    max_count = max(count)
    plt.figure(figsize=(10, 6))
    plt.xlim(-0.1, 1.1)
    plt.ylim(0, max_count)
    if max_count > 50:
        plt.yticks(np.append(np.arange(0, max_count + 2, 5), max_count))
    else:
        plt.yticks(np.arange(0, max_count + 2, 1))
    for i in range(len(count)):
        plt.vlines(x=i * interval, ymin=0, ymax=count[i], color='red', linestyle='-', linewidth=1)
    plt.axvline(x=mean_value, color='r', linestyle='-', linewidth=3, label='Mean')
    plt.hlines(y=max_count * 0.8, xmin=mean_value - std_dev, xmax=mean_value + std_dev, color='r', linestyle='-', linewidth=3)
    plt.axvline(x=F_opt, color='black', linestyle='--', linewidth=2, label='F_s')
    plt.xlabel('F', fontsize=18)
    plt.ylabel('#', fontsize=18)
    if std_dev == 0:
        z = 0
    else:
        z = (F_opt - mean_value)/std_dev
    if F_opt > max(values):
        if num_surros == 9:
            plt.title(f"z = {z:.6f} ; p = 0.1*", color='k', fontsize=24)
        elif num_surros == 19:
            plt.title(f"z = {z:.6f} ; p = 0.05**", color='k', fontsize=24)
        elif num_surros == 999:
            plt.title(f"z = {z:.6f} ; p = 0.001***", color='k', fontsize=24)
        else:
            p = 1/(num_surros+1)
            plt.title(f"z = {z:.6f} ; p = {p}" , color='k', fontsize=24)
    else:
        p = 1/(num_surros+1)
        plt.title(f"z = {z:.6f} ; p >> {p}", color='k', fontsize=24)
    plt.legend()

def plot_average_diagonal_value(STDM):
    """
    Plots the average value of each diagonal in the Spike Time Difference Matrix (STDM).

    :param STDM: The Spike Time Difference Matrix.
    :type STDM: numpy.ndarray

    Example::

            import matplotlib.pyplot as plt
            STDM = pyspike.Spike_time_difference_matrix(spike_trains)
            pyspike.plot_average_diagonal_value(STDM)
            plt.show()
    """
    num_trains = len(STDM)
    average_diagonal_values = []
    for i in range(num_trains):
        value = 0
        for j in range(num_trains-i):
                value += STDM[j][j+i]
        average_diagonal_values.append(value/(num_trains-i))

    plt.figure(figsize=(17, 10), dpi=80)
    plt.plot(average_diagonal_values, '-kx', linewidth=3, markersize=10, markeredgewidth=2)
    plt.title("Average diagonal value of STDM", color='k', fontsize=24)
    plt.xlabel('STDM value', fontsize=18)
    plt.ylabel('Average value', fontsize=18)
    plt.grid()

def plot_latency_correction(spikes, method=0):
    """
    Plots the spike trains before and after applying latency correction, along with the Spike Time Difference Matrix (STDM).
    
    :param spike_trains: List of spike trains.
    :type spike_trains: List of :class:`pyspike.SpikeTrain`
    :param method: The method to use for latency correction (0: simulated annealing, 1: extrapolation, 2: interpolation).
    :type method: int

    Example::

            import matplotlib.pyplot as plt
            pyspike.plot_latency_correction(spike_trains, method=0)
            plt.show()
    """
    tmin = spikes[0].t_start
    tmax = spikes[0].t_end
    num_trains = len(spikes)
    num_spikes = [len(i) for i in spikes]
    STDM = Spike_time_difference_matrix(spikes)
    if method == 0:
        try:
            from .cython.cython_simulated_annealing import sim_ann_latency_correction
        except ImportError:
            spk.NoCythonWarn()
        all_shifts = sim_ann_latency_correction(spikes)[0]
    elif method == 1:
        all_shifts = latency_correction_extrapol(STDM)[0]
    else:
        all_shifts = latency_correction_intrapol(STDM)[0]
    shifted_spikes = []
    for i in range(num_trains):
        for j in range(num_spikes[i]):
            shifted_spikes.append(spikes[i][j]-all_shifts[i])
    new_tmax = np.max(shifted_spikes)
    new_tmin = np.min(shifted_spikes)
    corrected_spikes1 = []
    N = 0
    for i in range(num_trains):
        L = []
        for j in range(num_spikes[i]):
            L.append((tmax-tmin)*(shifted_spikes[N]-new_tmin)/(new_tmax-new_tmin)+tmin)
            N += 1
        corrected_spikes1.append(L)
    for trc in range(num_trains-1):
        for trc2 in range(trc+1, num_trains):
            if num_spikes[trc]==num_spikes[trc2] and num_spikes[trc]>0 and max(abs(np.array(corrected_spikes1[trc])-np.array(corrected_spikes1[trc2])))<1e-12:
                corrected_spikes1[trc2]=corrected_spikes1[trc]
    corrected_spikes = []
    for i in corrected_spikes1:
        corrected_spikes.append(spk.SpikeTrain(i, [tmin, tmax]))
    fig = plt.figure(figsize=(17, 10), dpi=80)
    
    ax1 = fig.add_axes([0.1, 0.55, 0.7, 0.3])
    fig1, ax1, _, colorbar1, cax1 = plot_spike_trains(spikes, order_color=1, ax=ax1)
    ax1.set_title("Before multivariate latency correction", fontsize=24)
    ax1.set_xlabel("")
    colorbar1.remove()

    ax2 = fig.add_axes([0.1, 0.1, 0.7, 0.3])
    fig2, ax2, _, colorbar2, cax2 = plot_spike_trains(corrected_spikes, order_color=1, ax=ax2)
    ax2.set_title("After multivariate latency correction", fontsize=24)
    colorbar2.remove()
    fig2.colorbar(cax1, ax=[ax1, ax2], orientation='vertical', shrink=0.7)
    arrow_coords = []
    for i in range(num_trains):
        diff = (new_tmax-new_tmin)*(all_shifts[i]-tmin)/(tmax-tmin)+new_tmin
        if diff > 0.01*(tmax-tmin):
            arrow_coords.append(((tmin+tmax)/2+diff, num_trains-i, (tmin+tmax)/2, num_trains-i))
    for (x1, y1, x2, y2) in arrow_coords:
        ax2.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(facecolor='black', arrowstyle='->'))

    ax3 = fig.add_axes([0.7, 0.55, 0.25, 0.3])
    fig3, ax3, cax3, colorbar3 = plot_matrix(STDM, variable=6, ax=ax3)
    colorbar3.remove()
    ax3.set_xlabel("")
    ax3.set_ylabel("")

    corrected_STDM = Spike_time_difference_matrix(corrected_spikes)
    ax4 = fig.add_axes([0.7, 0.1, 0.25, 0.3])
    vmin = np.min(STDM)
    vmax = np.max(STDM)
    fig4, ax4, cax4, colorbar4 = plot_matrix(corrected_STDM, variable=6, ax=ax4, vmin=vmin, vmax=vmax)
    colorbar4.remove()
    ax4.set_ylabel("")
    ax4.set_title("")
    fig4.colorbar(cax4, ax=[cax4.axes, cax3.axes], orientation='vertical', shrink=0.7)