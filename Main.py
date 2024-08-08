import matplotlib.pyplot as plt
import pyspike as spk

measures                     =  0  # +1:ISI, +2:SPIKE, +4:RI-SPIKE, +8:SPIKE-Synchro, +16:SPIKE-Order, +32:Spike Train Order
showing                      =  0  # +1:spike trains, +2:distances, +4:profiles, +8:matrices
plotting                     =  0  # +1:spike trains, +2:distances, +4:profiles, +8:matrices
sorting                      =  0  # 0:Unsorted order, 1:Sorted order
latency_correction           =  0  # 0:none, +1:sim ann, +2:extrapolation, +4:intrapolation
num_surros                   =  0  # 0:none, >1:num_surros
spike_time_difference_matrix =  0  # 0:none, 1:STDM
matching                     =  0  # 0:none, 1:plot matching spikes
dataset  =  6
print("\n\n\ndataset: %3i" % (dataset))

if dataset == 1:
    tmin=0
    tmax=10
    spike_trains = []
    spike_trains.append(spk.SpikeTrain([0, 3, 7, 10], [tmin, tmax]))
    spike_trains.append(spk.SpikeTrain([0, 2, 6, 10], [tmin, tmax]))
    #spike_trains.append(spk.SpikeTrain([0, 4, 5, 8, 10], [tmin, tmax]))
    #spike_trains.append(spk.SpikeTrain([4, 5, 8], [tmin, tmax]))
elif dataset == 2:
    tmin=0
    tmax=100
    spike_trains = []
    spike_trains.append(spk.SpikeTrain([12, 16, 28, 32, 44, 48, 60, 64, 76, 80], [tmin, tmax]))
    spike_trains.append(spk.SpikeTrain([8, 20, 24, 36, 40, 52, 56, 68, 72, 84], [tmin, tmax]))
elif dataset == 3:
    tmin=0
    tmax=1000
    spike_trains = []
    spike_trains.append(spk.SpikeTrain([100, 200, 300], [tmin, tmax]))
    spike_trains.append(spk.SpikeTrain([105, 205, 305], [tmin, tmax]))
    spike_trains.append(spk.SpikeTrain([105, 195, 500], [tmin, tmax]))
elif dataset == 4:
    tmin=0
    tmax=1000
    spike_trains = spk.load_spike_trains_from_txt("./examples/PySpike_testdata4.txt", edges=(tmin, tmax))
elif dataset == 5:
    tmin=0
    tmax=1
    spike_trains = spk.load_spike_trains_from_txt("./examples/PySpike_testdata5.txt", edges=(tmin, tmax))
elif dataset == 6:
    tmin=0
    tmax=1000
    spike_trains = spk.load_spike_trains_from_txt("./examples/PySpike_testdata6.txt", edges=(tmin, tmax))
elif dataset == 40:
    tmin=0
    tmax=4000
    spike_trains = spk.load_spike_trains_from_txt("./examples/PySpike_testdata.txt", edges=(tmin, tmax))
elif dataset == 100:
    tmin=0
    tmax = 100 
    num_trains = 100
    num_synfire_events = 4 
    num_inverse_events = 0
    overlap = 0.4
    shuffle = 0.02
    jitter = 0.01
    complete = 0.9
    background = 1-complete
    order = 0
    plot = 0
    spike_trains = spk.create_synfire(tmin, tmax, num_trains, num_synfire_events, num_inverse_events, overlap, shuffle, jitter, complete, background, order, plot)[0]
num_trains = len(spike_trains)
print("\n\n\n\nnum_trains: %3i" % (num_trains))
spike_trains = spk.reconcile_spike_trains(spike_trains)

if showing % 2 > 0:                                                    # Spike trains
    spk.plot_spike_trains(spike_trains, showing=1)
if plotting % 2 > 0:
    spk.plot_spike_trains(spike_trains, order_color=1)

if measures % 2 > 0:                                                    # ISI-distance
    if showing % 16 > 1 or plotting % 16 > 1:
        isi_distance = spk.isi_distance(spike_trains)
        if showing % 4 > 1:
           print("\nISI-Distance: %.8f\n" % isi_distance)
    if showing % 8 > 3 or plotting % 8 > 3:
        isi_profile = spk.isi_profile(spike_trains)
        x, y = isi_profile.get_plottable_data()
        if showing % 8 > 3:
            spk.plot_profile(x, y, variable=1, variable_value=isi_distance, showing=1)
        if plotting % 8 > 3:
            spk.plot_profile(x, y, variable=1, variable_value=isi_distance)
    if showing % 16 > 7 or plotting % 16 > 7:
        isi_distance_mat = spk.isi_distance_matrix(spike_trains)
        if showing % 16 > 7:
            spk.plot_matrix(isi_distance_mat, variable=1, variable_value=isi_distance, showing=1)
        if plotting % 16 > 7:
            spk.plot_matrix(isi_distance_mat, variable=1, variable_value=isi_distance)

if measures % 4 > 1:                                                    # SPIKE-distance
    if showing % 16 > 1 or plotting % 16 > 1:
        spike_distance = spk.spike_distance(spike_trains)
        if showing % 4 > 1:
            print("\nSPIKE-Distance: %.8f\n" % spike_distance)
    if showing % 8 > 3 or plotting % 8 > 3:
        spike_profile = spk.spike_profile(spike_trains)
        x, y = spike_profile.get_plottable_data()
        if showing % 8 > 3:
            spk.plot_profile(x, y, variable=2, variable_value=spike_distance, showing=1)
        if plotting % 8 > 3:
            spk.plot_profile(x, y, variable=2, variable_value=spike_distance)
    if showing % 16 > 7 or plotting % 16 > 7:
        spike_distance_mat = spk.spike_distance_matrix(spike_trains)
        if showing % 16 > 7:
            spk.plot_matrix(spike_distance_mat, variable=2, variable_value=spike_distance, showing=1)
        if plotting % 16 > 7:
            spk.plot_matrix(spike_distance_mat, variable=2, variable_value=spike_distance)

if measures % 8 > 3:                                                    # RI-SPIKE-distance
    if showing % 16 > 1 or plotting % 16 > 1:
        ri_spike_distance = spk.spike_distance(spike_trains, RI=True)
        if showing % 4 > 1:
            print("\nRI-SPIKE-Distance: %.8f\n" % ri_spike_distance)
    if showing % 8 > 3 or plotting % 8 > 3:
        ri_spike_profile = spk.spike_profile(spike_trains, RI=True)
        x, y = ri_spike_profile.get_plottable_data()
        if showing % 8 > 3:
            spk.plot_profile(x, y, variable=2, variable_value=ri_spike_distance, showing=1)
        if plotting % 8 > 3:
            spk.plot_profile(x, y, variable=2, variable_value=ri_spike_distance)
    if showing % 16 > 7 or plotting % 16 > 7:
        ri_spike_distance_mat = spk.spike_distance_matrix(spike_trains, RI=True)
        if showing % 16 > 7:
            spk.plot_matrix(ri_spike_distance_mat, variable=2, variable_value=ri_spike_distance, showing=1)
        if plotting % 16 > 7:
            spk.plot_matrix(ri_spike_distance_mat, variable=2, variable_value=ri_spike_distance)

if measures % 16 > 7:                                                    # Spike-Synchro
    if showing % 16 > 1 or plotting % 16 > 1:
        spike_synchro = spk.spike_sync(spike_trains)
        if showing % 4 > 1:
            print("\nSpike-Synchro: %.8f\n" % spike_synchro)
    if showing % 8 > 3 or plotting % 8 > 3:
        spike_sync_profile = spk.spike_sync_profile(spike_trains)
        x, y = spike_sync_profile.get_plottable_data()
        if showing % 8 > 3:
            spk.plot_profile(x, y, variable=3, variable_value=spike_synchro, showing=1)
        if plotting % 8 > 3:
            spk.plot_profile(x, y, variable=3, variable_value=spike_synchro)
    if showing % 16 > 7 or plotting % 16 > 7:
        spike_synchro_mat = spk.spike_sync_matrix(spike_trains)
        if showing % 16 > 7:
            spk.plot_matrix(spike_synchro_mat, variable=3, variable_value=spike_synchro, showing=1)
        if plotting % 16 > 7:
            spk.plot_matrix(spike_synchro_mat, variable=3, variable_value=spike_synchro)

if  measures % 32 > 15:                                                   #SPIKE-Order
    if showing % 16 > 1 or plotting % 16 > 1:
        SPIKE_Order = 0
        if showing % 4 > 1:
            print("\nSPIKE-Order: %.8f, by definition\n" % SPIKE_Order)
    if showing % 8 > 3 or plotting % 8 > 3:
        x, y = spk.spike_order_profile(spike_trains)
        if showing % 8 > 3:
            spk.plot_profile(x, y, variable=4, variable_value=SPIKE_Order, showing=1)
        if plotting % 8 > 3:
            spk.plot_profile(x, y, variable=4, variable_value=SPIKE_Order)
    if showing % 16 > 7 or plotting % 16 > 7:
        spike_order_mat = spk.spike_order_matrix(spike_trains)
        if showing % 8 > 3:
            spk.plot_matrix(spike_order_mat, variable=4, variable_value=SPIKE_Order, showing=1)
        if plotting % 16 > 7:
            spk.plot_matrix(spike_order_mat, variable=4, variable_value=SPIKE_Order)

if  measures % 64 > 31:                                                   #Spike train order
     if showing % 16 > 1 or plotting % 16 > 1:
        spike_train_order = spk.spike_train_order(spike_trains)
        if showing % 4 > 1:
            print("\nSpike train order: %.8f\n" % spike_train_order)
     if showing % 8 > 3 or plotting % 8 > 3:
        spike_order_profile = spk.spike_train_order_profile(spike_trains)
        x, y = spike_order_profile.get_plottable_data()
        if showing % 8 > 3:
            spk.plot_profile(x, y, variable=5, variable_value=spike_train_order, showing=1)
        if plotting % 8 > 3:
            spk.plot_profile(x, y, variable=5, variable_value=spike_train_order)
     if showing % 16 > 7 or plotting % 16 > 7:
        spike_train_order_mat = spk.spike_train_order_matrix(spike_trains)
        if showing % 16 > 7:
            spk.plot_matrix(spike_train_order_mat, variable=5, variable_value=spike_train_order, showing=1)
        if plotting % 16 > 7:
            spk.plot_matrix(spike_train_order_mat, variable=5, variable_value=spike_train_order)

if sorting % 2 > 0:                                                       #Sorting
    print("The first print or plotting will be always the normal one and the second will be the sorted one")
    E_init = spk.spike_train_order_matrix(spike_trains)
    F_init = spk.spike_train_order(spike_trains)
    phi, _ = spk.optimal_spike_train_sorting(spike_trains)
    E_opt = spk.permutate_matrix(E_init, phi)
    F_opt = spk.spike_train_order(spike_trains, indices=phi)
    opt_spike_trains = []
    for i in range(num_trains):
        L = []
        for j in range(len(spike_trains[phi[i]])):
            L.append(spike_trains[phi[i]][j])
        opt_spike_trains.append(spk.SpikeTrain(L, [tmin, tmax]))

    if showing % 2 > 0 or plotting % 2 > 0:
        if showing % 2 > 0:
            spk.plot_spike_trains(spike_trains, phi=phi, showing=1)
        if plotting % 2 > 0:
            spk.plot_spike_trains(spike_trains, phi=phi, order_color=1)
    if showing % 8 > 3 or plotting % 8 > 3:
        if showing % 8 > 3:
            spike_train_order = spk.spike_train_order_value(spike_trains)
            opt_spike_order_profile = spk.spike_train_order_profile(spike_trains)
            x, y = opt_spike_order_profile.get_plottable_data()
            spk.plot_profile(x, y, variable=5, variable_value=spike_train_order, showing=1)

            spike_train_order = spk.spike_train_order_value(opt_spike_trains)
            opt_spike_order_profile = spk.spike_train_order_profile(opt_spike_trains)
            x, y = opt_spike_order_profile.get_plottable_data()
            spk.plot_profile(x, y, variable=5, variable_value=spike_train_order, showing=1)

        if plotting % 8 > 3:
            spike_train_order = spk.spike_train_order_value(spike_trains)
            opt_spike_order_profile = spk.spike_train_order_profile(spike_trains)
            x, y = opt_spike_order_profile.get_plottable_data()
            spk.plot_profile(x, y, variable=5, variable_value=spike_train_order)

            spike_train_order = spk.spike_train_order_value(opt_spike_trains)
            opt_spike_order_profile = spk.spike_train_order_profile(opt_spike_trains)
            x, y = opt_spike_order_profile.get_plottable_data()
            spk.plot_profile(x, y, variable=5, variable_value=spike_train_order)

    if showing % 16 > 7 or plotting % 16 > 7:
        if showing % 16 > 7:
            print("E_init:\n")
            print(E_init,"\n\n")
            print("E_opt:\n")
            print(E_opt,"\n\n")
            print("Synfire Indicator of original spike trains:", F_init)
            print("Synfire Indicator of optimized spike train sorting:", F_opt)

        if plotting % 16 > 7:
            spk.plot_matrix(E_init, variable=5, variable_value=F_init)
            _, ax, _, _ = spk.plot_matrix(E_opt, variable=5, variable_value=F_opt)
            ax.set_title(f"Sorted Spike train order Matrix (Spike train order = {F_opt})", color='k', fontsize=24)

if latency_correction % 2 > 0:                                                      #latency correction
    spk.plot_latency_correction(spike_trains, method=0)
if latency_correction % 4 > 1:
    spk.plot_latency_correction(spike_trains, method=1)
if latency_correction % 8 > 3:
    spk.plot_latency_correction(spike_trains, method=2)

if num_surros > 0:                                                               #surrogate
    spk.plot_surrogates(spike_trains, num_surros)

if spike_time_difference_matrix % 2 > 0:                                         #spike_time_difference_matrix
    STDM = spk.Spike_time_difference_matrix(spike_trains)
    spk.plot_matrix(STDM, variable=6, showing=1)
    spk.plot_matrix(STDM, variable=6)
    spk.plot_average_diagonal_value(STDM)

if matching % 2 > 0:                                                             #matching
    spk.spike_matching_plot(spike_trains)

plt.show()
plt.close()

