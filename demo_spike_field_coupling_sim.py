import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.signal
import seaborn as sns


def plot_circular_phase_plot(ax_inset, all_phases):

    # calculate mean resultant vector
    all_phases = np.hstack(all_phases)
    mean_r = np.sum(np.exp(1j * all_phases)) / len(all_phases)

    ax_inset.arrow(
        np.angle(mean_r),
        0,
        0,
        np.abs(mean_r),
        alpha=1,
        width=0.03,
        length_includes_head=True,
        head_width=0.1,
        edgecolor="#FF5733",
        facecolor="#FF5733",
        lw=1,
        zorder=5,
    )

    # make polar plot prettier
    ax_inset.set_theta_zero_location("N")
    ax_inset.set_ylim(0, 1.1)
    ax_inset.grid(False)
    ax_inset.set_xticks([])
    ax_inset.set_yticklabels([])
    ax_inset.text(0, 0.4 * np.pi, "LFP phases\n at spike times", ha="center")


# set up plot
fig = plt.figure(figsize=(7.4, 4))
gs = gridspec.GridSpec(3, 1, height_ratios=[1, 2, 1], hspace=0.5, bottom=0.15)

# parameters
nr_units = 25
firing_rate = 10
dt = 1 / 1000
nr_seconds = 5
time = np.arange(0, nr_seconds, dt)
nr_conditions = 10

# morphing factor 1: homogeneous -> 0: inhomogeneous
factors = np.linspace(1, 0, nr_conditions)

# modulating function for inhomogeneous process, maximum spiking at trough = pi
phase_shift = np.pi
r_t = np.sin(2 * np.pi * (time - 0.25) - phase_shift)

# some random matrix for generating Poisson spiking
randomness = np.random.rand(len(time), nr_units)

# plot LFP trace
lfp = np.sin(2 * np.pi * (time - 0.25))
phases = np.angle(scipy.signal.hilbert(lfp))

ax_lfp = fig.add_subplot(gs[0, 0])
ax_lfp.plot(time, lfp, color="#4B0082", lw=2)
ax_lfp.set(
    yticks=[],
    xlim=(time[0], time[-1]),
    title="spike-field coupling",
    ylabel="LFP",
)
sns.despine(ax=ax_lfp, left=True)

# create axes: unit spiking activity
ax_units = fig.add_subplot(gs[1, 0])
ax_units.set(
    ylabel="spikes\n single units",
    xlim=(time[0], time[-1]),
    yticks=[],
)
sns.despine(ax=ax_units)

# create axes: histogram for all spikes
bins = np.arange(0, nr_seconds, 0.05)
ax_hist = fig.add_subplot(gs[2, 0])
ax_hist.set(
    yticks=[],
    xlim=(time[0], time[-1]),
    xlabel="time [s]",
    ylabel="spikes\nall units",
    ylim=(0, nr_units + 1),
)
sns.despine(ax=ax_hist, left=True)

# create small inset axes: circular phase plot
ax_inset = fig.add_axes([0.8, 0.65, 0.20, 0.20], projection="polar")

for factor in factors:

    ax_inset.cla()
    lines = []
    all_spikes = []
    all_phases = []

    # define weighting of time dependent rate function
    r = dt * (1 - factor) * r_t + dt

    # normalize area so the firing rate stays ~the same for all conditions
    r[r < 0] = 0
    r = r / np.trapz(r) * firing_rate * nr_seconds

    for i in range(nr_units):

        # generate Poisson spike times
        spikes = randomness[:, i] < r
        spike_times = time[spikes]

        # plot unit activity
        line = ax_units.vlines(spike_times, ymin=i, ymax=i + 1, color="k")
        lines.append(line)

        # plot LFP phase at each spike
        lfp_phases = phases[spikes]
        for phase in lfp_phases:
            ax_inset.arrow(phase, 0, 0, 1, color="k", alpha=0.05)

        # collect information from all units
        all_spikes.append(spike_times)
        all_phases.append(lfp_phases)

    # calculate mean resultant vector
    plot_circular_phase_plot(ax_inset, all_phases)

    # plot histogram across all units
    all_spikes = np.hstack(all_spikes)
    histogram = ax_hist.hist(all_spikes, bins, color="tab:red")

    plt.pause(0.1)
    [line.remove() for line in lines]
    [h.remove() for h in histogram[2]]

fig.show()
