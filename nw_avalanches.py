# nw_avalanches.py
# Avalanches from the list of switching times, using the same binning as
# Hochstetter et al. (DOI 10.1038/s41467-021-24260-z):
#
#   1. Frame width = mean interval between consecutive switches.
#   2. An avalanche = a run of non-empty frames between two empty frames.
#   3. Size S = number of switches in the run, duration T = number of frames.
#
# Example: times = [0, 0.5, 1.0, 1.1, 3.0, 3.2, 5.0]
#   mean interval = 5.0 / 6 = 0.8333, frame k = [k*0.8333, (k+1)*0.8333)
#   frames of the 7 events: 0, 0, 1, 1, 3, 3, 6
#   counts per frame = [2, 2, 0, 2, 0, 0, 1]
#   avalanches: frames 0-1 (S=4, T=2), frame 3 (S=2, T=1), frame 6 (S=1, T=1)

import numpy as np


def find_avalanches(event_time):
    # Returns a dict: frame_width, counts (switches per frame),
    # S (sizes) and T (durations), one entry per avalanche.
    et = np.asarray(event_time, dtype=float)
    n = et.shape[0]

    result = {}
    if n < 2:
        # Not enough events to define a frame width.
        result["frame_width"] = 0.0
        result["counts"] = np.array([n], dtype=int)
        result["S"] = np.array([n], dtype=int)
        result["T"] = np.array([1], dtype=int)
        if n == 0:
            result["counts"] = np.array([], dtype=int)
            result["S"] = np.array([], dtype=int)
            result["T"] = np.array([], dtype=int)
        return result

    width = (et[-1] - et[0]) / (n - 1)
    if width <= 0.0:
        # All switches at the same clock: one avalanche, one frame.
        result["frame_width"] = 0.0
        result["counts"] = np.array([n], dtype=int)
        result["S"] = np.array([n], dtype=int)
        result["T"] = np.array([1], dtype=int)
        return result

    # Frame index of every event. The small 1e-9 keeps the last event
    # (which sits exactly on a frame edge) from falling into the previous
    # frame because of rounding.
    idx = np.floor((et - et[0]) / width + 1e-9).astype(int)
    counts = np.bincount(idx)

    sizes = []
    durations = []
    run_size = 0
    run_frames = 0
    for c in counts:
        if c > 0:
            run_size = run_size + int(c)
            run_frames = run_frames + 1
        else:
            if run_frames > 0:
                sizes.append(run_size)
                durations.append(run_frames)
            run_size = 0
            run_frames = 0
    if run_frames > 0:
        sizes.append(run_size)
        durations.append(run_frames)

    result["frame_width"] = width
    result["counts"] = counts
    result["S"] = np.array(sizes, dtype=int)
    result["T"] = np.array(durations, dtype=int)
    return result
