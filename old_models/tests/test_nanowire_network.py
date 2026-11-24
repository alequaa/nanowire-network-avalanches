# test_nanowire_network.py
# Checks for nanowire_network.py. Run: python tests/test_nanowire_network.py
# Each check prints OK or stops with an error.

import os
import sys

# The library files (nanowire_network.py, nw_solver.py, ...) are in the main
# folder, one level up: add it to the search path.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import nanowire_network as nn

# Figures go in an 'images' folder next to this script, no matter where the
# terminal is open.
IMAGES_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")


def test_hand_crossings():
    # Wires 0,1: (0,0)-(2,2) and (0,2)-(2,0) -> cross at (1,1).
    # Wire 2: isolated.
    # Wires 3,4: would cross only if wire 4 were extended -> must NOT count.
    # Wires 5,6: parallel -> must NOT count.
    p1 = np.array([[0, 0], [0, 2], [5, 5], [3, 0], [4, 0.5], [10, 0], [10, 1]], float)
    p2 = np.array([[2, 2], [2, 0], [6, 5], [3, 1], [5, 0.5], [12, 0], [12, 1]], float)
    jn, xy = nn.find_crossings(p1, p2)
    assert jn.shape[0] == 1, "expected 1 junction, found " + str(jn.shape[0])
    assert tuple(jn[0]) == (0, 1)
    assert np.allclose(xy[0], [1.0, 1.0])
    print("OK test 1: hand-made crossings")


# Slow independent method for test 2: orientation (cross product signs).
def orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def slow_cross(a1, a2, b1, b2):
    d1 = orient(a1, a2, b1)
    d2 = orient(a1, a2, b2)
    d3 = orient(b1, b2, a1)
    d4 = orient(b1, b2, a2)
    if d1 * d2 < 0 and d3 * d4 < 0:
        return True
    else:
        return False


def test_against_slow_method():
    for seed in range(5):
        net = nn.generate_network(n_wires=60, seed=seed, clip_to_area=False,
                                  keep_largest=False)
        found = set()
        for k in range(net.n_junctions()):
            found.add((int(net.junctions[k, 0]), int(net.junctions[k, 1])))
        slow = set()
        n = net.n_wires()
        for i in range(n):
            for j in range(i + 1, n):
                if slow_cross(net.p1[i], net.p2[i], net.p1[j], net.p2[j]):
                    slow.add((i, j))
        assert found == slow, "seed " + str(seed) + ": methods differ"
    print("OK test 2: same junctions as the slow method on 5 networks of 60 wires")


def test_components():
    # Nodes 0,1,2 together; 3,4 together; 5 alone.
    edges = np.array([[0, 1], [1, 2], [3, 4]])
    label = nn.connected_components(6, edges)
    assert label[0] == label[1] == label[2]
    assert label[3] == label[4]
    assert len(set(label.tolist())) == 3
    print("OK test 3: connected components by hand")


def test_largest_component_consistency():
    # After keep_largest: connected, valid indices, each junction lies on
    # both of its wires.
    for seed in range(10):
        net = nn.generate_network(seed=seed)
        label = nn.connected_components(net.n_wires(), net.junctions)
        assert label.max() == 0, "network is not connected"
        assert net.junctions.min() >= 0
        assert net.junctions.max() < net.n_wires()
        for k in range(net.n_junctions()):
            i = net.junctions[k, 0]
            j = net.junctions[k, 1]
            x = net.junction_xy[k]
            for w in (i, j):
                d = net.p2[w] - net.p1[w]
                t = np.dot(x - net.p1[w], d) / np.dot(d, d)
                dist = np.linalg.norm(net.p1[w] + t * d - x)
                assert dist < 1e-8 and -1e-9 <= t <= 1 + 1e-9
    print("OK test 4: single component, valid indices, junctions on the right wires")


def test_length_distribution():
    # Same gamma formula as the generator, many samples (mean 10, std 1).
    rng = np.random.default_rng(1)
    shape = (10.0 / 1.0) ** 2
    scale = 1.0 ** 2 / 10.0
    L = rng.gamma(shape, scale, size=200000)
    assert abs(L.mean() - 10.0) < 0.02 and abs(L.std() - 1.0) < 0.02
    print("OK test 5: gamma mean", round(L.mean(), 3), "std", round(L.std(), 3))


def test_electrodes():
    net = nn.generate_network(seed=3)
    s, d = nn.point_electrodes(net)
    assert len(s) == 1 and len(d) == 1 and s[0] != d[0]
    big = nn.generate_network(n_wires=2250, side=150.0, seed=0)
    ss, dd = nn.strip_electrodes(big, 0.025)
    assert len(set(ss) & set(dd)) == 0
    print("OK test 6: point electrodes; strips on 2250 wires / 150 um:",
          len(ss), "sources,", len(dd), "drains,", big.n_junctions(), "junctions")


def statistics():
    # Not pass/fail: numbers to look at.
    for clip in (True, False):
        nj = []
        nw = []
        for seed in range(300):
            net = nn.generate_network(seed=seed, clip_to_area=clip)
            nj.append(net.n_junctions())
            nw.append(net.n_wires())
        nj = np.array(nj)
        nw = np.array(nw)
        print("clip_to_area =", clip, "| 300 networks, 100 wires thrown")
        print("   wires in largest component: mean", round(nw.mean(), 1),
              " min", nw.min(), " max", nw.max())
        print("   junctions: mean", round(nj.mean(), 1), " std", round(nj.std(), 1),
              " min", nj.min(), " max", nj.max())


if __name__ == "__main__":
    test_hand_crossings()
    test_against_slow_method()
    test_components()
    test_largest_component_consistency()
    test_length_distribution()
    test_electrodes()
    statistics()
    net = nn.generate_network(seed=7)
    s, d = nn.point_electrodes(net)
    os.makedirs(IMAGES_FOLDER, exist_ok=True)
    figure_path = os.path.join(IMAGES_FOLDER, "example_network.png")
    nn.plot_network(net, s, d, figure_path)
    print("Figure saved in:", figure_path)
