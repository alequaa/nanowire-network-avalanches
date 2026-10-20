# nanowire_network.py
# Random nanowire network generator (Hochstetter et al., Nat. Commun. 12, 4008,
# DOI 10.1038/s41467-021-24260-z).
#
# Steps: random segments -> crossings (junctions) -> graph -> largest
# connected component -> electrodes.
#
# Example: wire 0 from (0,0) to (2,2), wire 1 from (0,2) to (2,0).
# They cross at (1,1): 1 junction, the pair (0, 1).

import numpy as np


class NanowireNetwork:
    # Plain container for everything that describes the network.
    def __init__(self):
        self.side = 0.0            # side of the square (micrometres)
        self.centers = None        # (n,2) wire centres
        self.lengths = None        # (n,) wire lengths
        self.angles = None         # (n,) wire angles (radians)
        self.p1 = None             # (n,2) first endpoint of each wire
        self.p2 = None             # (n,2) second endpoint of each wire
        self.junctions = None      # (m,2) pairs (i,j): wire i crosses wire j
        self.junction_xy = None    # (m,2) position of each crossing

    def n_wires(self):
        return self.centers.shape[0]

    def n_junctions(self):
        return self.junctions.shape[0]


def endpoints_from_center(centers, lengths, angles):
    # Endpoints = centre -/+ half the wire vector.
    # Example: centre (10,10), length 10, angle 0 -> (5,10) and (15,10).
    half_x = 0.5 * lengths * np.cos(angles)
    half_y = 0.5 * lengths * np.sin(angles)
    half = np.stack([half_x, half_y], axis=1)
    p1 = centers - half
    p2 = centers + half
    return p1, p2


def find_crossings(p1, p2):
    # Segment A = p + t*r, segment B = q + u*s, with t and u in [0,1].
    # With cross(a,b) = ax*by - ay*bx:
    #   denom = cross(r,s), t = cross(q-p, s)/denom, u = cross(q-p, r)/denom
    # They cross if denom != 0 and both t and u are in [0,1].
    # Example: A (0,0)-(2,2), B (0,2)-(2,0): denom = -8, t = u = 0.5,
    # crossing point p + 0.5*r = (1,1).
    n = p1.shape[0]
    r = p2 - p1

    # All pairs at once: [:, None] runs over i, [None, :] over j.
    px = p1[:, 0][:, None]
    py = p1[:, 1][:, None]
    rx = r[:, 0][:, None]
    ry = r[:, 1][:, None]
    qx = p1[:, 0][None, :]
    qy = p1[:, 1][None, :]
    sx = r[:, 0][None, :]
    sy = r[:, 1][None, :]

    denom = rx * sy - ry * sx
    dx = qx - px
    dy = qy - py

    # Parallel wires (denom = 0) are discarded; dividing by 1 avoids 0/0.
    parallel = (denom == 0.0)
    safe_denom = np.where(parallel, 1.0, denom)
    t = (dx * sy - dy * sx) / safe_denom
    u = (dx * ry - dy * rx) / safe_denom

    inside = (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)
    # Only i < j: each pair counted once, no wire crosses itself.
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    mask = inside & (~parallel) & upper

    ii, jj = np.nonzero(mask)
    junctions = np.stack([ii, jj], axis=1)
    xy_x = p1[ii, 0] + t[ii, jj] * r[ii, 0]
    xy_y = p1[ii, 1] + t[ii, jj] * r[ii, 1]
    junction_xy = np.stack([xy_x, xy_y], axis=1)
    return junctions, junction_xy


def connected_components(n_nodes, edges):
    # Breadth-first search. Returns label[i] = component number of node i.
    # Example: 5 nodes, edges (0,1) (1,2) (3,4) -> label = [0,0,0,1,1].
    neighbors = []
    for i in range(n_nodes):
        neighbors.append([])
    for k in range(edges.shape[0]):
        a = int(edges[k, 0])
        b = int(edges[k, 1])
        neighbors[a].append(b)
        neighbors[b].append(a)

    label = -np.ones(n_nodes, dtype=int)   # -1 = not visited yet
    current = 0
    for start in range(n_nodes):
        if label[start] != -1:
            continue
        label[start] = current
        queue = [start]
        head = 0
        while head < len(queue):
            node = queue[head]
            head = head + 1
            for nb in neighbors[node]:
                if label[nb] == -1:
                    label[nb] = current
                    queue.append(nb)
        current = current + 1
    return label


def keep_largest_component(net):
    # Keep only the wires of the biggest component and renumber from 0.
    n = net.n_wires()
    label = connected_components(n, net.junctions)
    sizes = np.bincount(label)
    biggest = int(np.argmax(sizes))
    keep = (label == biggest)

    # new_index[old] = new index (-1 if the wire is dropped)
    new_index = -np.ones(n, dtype=int)
    new_index[keep] = np.arange(int(keep.sum()))

    jn = net.junctions
    jn_keep = keep[jn[:, 0]] & keep[jn[:, 1]]

    net.centers = net.centers[keep]
    net.lengths = net.lengths[keep]
    net.angles = net.angles[keep]
    net.p1 = net.p1[keep]
    net.p2 = net.p2[keep]
    net.junction_xy = net.junction_xy[jn_keep]
    net.junctions = new_index[jn[jn_keep]]
    return net


def generate_network(n_wires=100, side=30.0, mean_len=10.0, std_len=1.0,
                     seed=None, clip_to_area=True, keep_largest=True):
    # Defaults = Hochstetter's standard network (lengths in micrometres).
    # clip_to_area: count a crossing only if it lies inside the square.
    #   The paper does not say which choice it uses (unverified).
    # Gamma lengths: shape k = (mean/std)^2, scale = std^2/mean.
    #   Example: mean 10, std 1 -> k = 100, scale = 0.1 (k*scale = 10).
    rng = np.random.default_rng(seed)

    shape = (mean_len / std_len) ** 2
    scale = (std_len ** 2) / mean_len
    lengths = rng.gamma(shape, scale, size=n_wires)
    centers = rng.uniform(0.0, side, size=(n_wires, 2))
    angles = rng.uniform(0.0, np.pi, size=n_wires)

    p1, p2 = endpoints_from_center(centers, lengths, angles)
    junctions, junction_xy = find_crossings(p1, p2)

    if clip_to_area:
        x_ok = (junction_xy[:, 0] >= 0.0) & (junction_xy[:, 0] <= side)
        y_ok = (junction_xy[:, 1] >= 0.0) & (junction_xy[:, 1] <= side)
        ok = x_ok & y_ok
        junctions = junctions[ok]
        junction_xy = junction_xy[ok]

    net = NanowireNetwork()
    net.side = side
    net.centers = centers
    net.lengths = lengths
    net.angles = angles
    net.p1 = p1
    net.p2 = p2
    net.junctions = junctions
    net.junction_xy = junction_xy

    if keep_largest:
        net = keep_largest_component(net)
    return net


def point_electrodes(net):
    # Case A: one source wire and one drain wire at opposite ends.
    # My choice (the paper only shows two markers): source = highest
    # centre, drain = lowest centre.
    y = net.centers[:, 1]
    source = int(np.argmax(y))
    drain = int(np.argmin(y))
    return [source], [drain]


def strip_electrodes(net, fraction=0.025):
    # Case B: all wires with centre within fraction*side of the top edge
    # are sources, of the bottom edge are drains (Hochstetter Figs 4-5
    # use 2.5%). Example: side 150 -> strip thickness 3.75.
    y = net.centers[:, 1]
    thickness = fraction * net.side
    sources = []
    drains = []
    for i in range(net.n_wires()):
        if y[i] >= net.side - thickness:
            sources.append(i)
        if y[i] <= thickness:
            drains.append(i)
    return sources, drains


def plot_network(net, sources=None, drains=None, filename=None):
    # Draws the network and saves it as an image file (filename, e.g. "net.png").
    # Wires grey, junctions blue, sources green, drains red.
    # If filename is None nothing is saved (and no window is opened).
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    for i in range(net.n_wires()):
        ax.plot([net.p1[i, 0], net.p2[i, 0]],
                [net.p1[i, 1], net.p2[i, 1]],
                color="0.35", linewidth=0.8)
    ax.plot(net.junction_xy[:, 0], net.junction_xy[:, 1], "o",
            color="tab:blue", markersize=2.5)
    if sources is not None:
        for i in sources:
            ax.plot([net.p1[i, 0], net.p2[i, 0]],
                    [net.p1[i, 1], net.p2[i, 1]],
                    color="tab:green", linewidth=2.2)
    if drains is not None:
        for i in drains:
            ax.plot([net.p1[i, 0], net.p2[i, 0]],
                    [net.p1[i, 1], net.p2[i, 1]],
                    color="tab:red", linewidth=2.2)
    ax.plot([0, net.side, net.side, 0, 0], [0, 0, net.side, net.side, 0],
            color="k", linewidth=0.6, linestyle="--")
    ax.set_aspect("equal")
    ax.set_xlabel("x (micrometres)")
    ax.set_ylabel("y (micrometres)")
    ax.set_title(str(net.n_wires()) + " nanowires, " +
                 str(net.n_junctions()) + " junctions")
    if filename is not None:
        fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
