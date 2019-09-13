import numpy as np
import matplotlib.pyplot as plt
from src.cfr import CFR


def fd(energy, ef, temp):
    """
    Fermi-Dirac function

    :param energy:   energy in eV
    :param ef:       Fermi level in eV
    :param temp:     temperature in K
    :return:         numpy.ndarray of the FD distribution
    """

    kb = 8.61733e-5  # Boltzmann constant in eV
    return 1.0 / (1.0 + np.exp((energy - ef) / (kb * temp)))


def t_order_frac(x):
    return 0.5 * (np.sign(x) + 1.0) / x


def poles_and_residues_cutoff_50():
    """
    A table fo precomputed positions of poles and their residuals for the Fermi-Dirac function

    :param cutoff:   cutoff energy
    :return:
    """

    poles = np.array([-0.31830989 + 0.j, 0.31830989 + 0.j, -0.1061033 + 0.j, 0.1061033 + 0.j,
                      -0.06366198 + 0.j, 0.06366198 + 0.j, -0.04547284 + 0.j, 0.04547284 + 0.j,
                      -0.03536777 + 0.j, 0.03536777 + 0.j, -0.02893726 + 0.j, 0.02893726 + 0.j,
                      -0.02448538 + 0.j, 0.02448538 + 0.j, -0.02122066 + 0.j, 0.02122066 + 0.j,
                      -0.01872411 + 0.j, -0.01675315 + 0.j, 0.01872411 + 0.j, -0.01515761 + 0.j,
                      -0.01383956 + 0.j, 0.01675315 + 0.j, -0.01273239 + 0.j, -0.01178907 + 0.j,
                      -0.01097099 + 0.j, -0.01021542 + 0.j, -0.00942084 + 0.j, -0.00853262 + 0.j,
                      -0.00755524 + 0.j, 0.01515761 + 0.j, -0.00650519 + 0.j, 0.01383956 + 0.j,
                      -0.00539707 + 0.j, -0.00424334 + 0.j, -0.0030551 + 0.j, -0.00184263 + 0.j,
                      0.01273239 + 0.j, -0.0006158 + 0.j, 0.01178907 + 0.j, 0.01097099 + 0.j,
                      0.0006158 + 0.j, 0.00184263 + 0.j, 0.01021542 + 0.j, 0.00942084 + 0.j,
                      0.00853262 + 0.j, 0.0030551 + 0.j, 0.00755524 + 0.j, 0.00650519 + 0.j,
                      0.00424334 + 0.j, 0.00539707 + 0.j])

    residues = np.array([1. + 0.j, 1. + 0.j, 1. + 0.j,
                         1. + 0.j, 1. + 0.j, 1. + 0.j,
                         1. + 0.j, 1. + 0.j, 1. + 0.j,
                         1. + 0.j, 1. + 0.j, 1. + 0.j,
                         1. + 0.j, 1. + 0.j, 1. + 0.j,
                         1. + 0.j, 1. + 0.j, 1. + 0.j,
                         1. + 0.j, 1. + 0.j, 1.00000005 + 0.j,
                         1. + 0.j, 1.00001109 + 0.j, 1.00083631 + 0.j,
                         1.02042845 + 0.j, 1.15734258 + 0.j, 1.5049063 + 0.j,
                         2.04504351 + 0.j, 2.83386414 + 0.j, 1. + 0.j,
                         4.06677965 + 0.j, 1.00000005 + 0.j, 6.18984564 + 0.j,
                         10.36600881 + 0.j, 20.49667166 + 0.j, 57.2467533 + 0.j,
                         1.00001109 + 0.j, 516.57150851 + 0.j, 1.00083631 + 0.j,
                         1.02042845 + 0.j, 516.57150851 + 0.j, 57.2467533 + 0.j,
                         1.15734258 + 0.j, 1.5049063 + 0.j, 2.04504351 + 0.j,
                         20.49667166 + 0.j, 2.83386414 + 0.j, 4.06677965 + 0.j,
                         10.36600881 + 0.j, 6.18984564 + 0.j])

    return poles, residues





def test_gf1(z=1j * 1e-3):
    return t_order_frac(z + 10.0) + \
           t_order_frac(z + 5.0) + \
           t_order_frac(z + 2.0) + \
           t_order_frac(z - 5.0)


def test_gf(z=1j * 1e-3):
    return 1.0 / (z + 10.0) + \
           1.0 / (z + 5.0) + \
           1.0 / (z + 2.0) + \
           1.0 / (z - 5.0)


def test_itegrate():
    fd = CFR(100)
    Ef = np.linspace(-12, 12, 150)
    ans = []

    for ef in Ef:
        ans.append(fd.integrate(test_gf, ef=ef, tempr=600, t_ordered=False))

    return np.array(ans)


def test_itegrate1():
    tempr = 600
    # fd_poles_coords, fd_poles = poles_and_residues(cutoff=40)
    # fd_poles_coords, fd_poles = poles_and_residues_cutoff_50()
    fd = CFR(50)
    fd_poles_coords, fd_poles = fd.fd_poles_coords, fd.fd_poles

    Ef = np.linspace(-12, 12, 150)
    ans = []
    R = 1.0e10
    moment = 1j * R * test_gf(1j * R)

    for ef in Ef:
        points = fd.genetate_integration_points(ef, tempr)
        gf_vals = test_gf(points)
        ans.append(fd.integrate1(gf_vals, tempr, zero_moment=moment))

    return np.array(ans)


def bf_integration(Ef, tempr, gf=test_gf):
    # temp = 100
    R = 2e2
    # Ef = 2.1

    energy = np.linspace(-R + Ef, R + Ef, 3e7)
    ans = np.imag(np.trapz(test_gf(energy + 1j * 10e-5) * fd(energy, Ef, tempr), energy))

    return -2 / np.pi * ans


def approximant(energy, poles, residues):
    arg = np.array(energy)
    ans = np.zeros(arg.shape) + 0.5

    for j in range(len(poles)):
        if poles[j] > 0:
            ans = ans - residues[j] / (arg - 1j / poles[j]) - residues[j] / (arg + 1j / poles[j])

    return ans


def test_approximation():
    a1, b1 = CFR(2).get_poles_and_residues()
    a2, b2 = CFR(10).get_poles_and_residues()
    a3, b3 = CFR(30).get_poles_and_residues()
    a4, b4 = CFR(50).get_poles_and_residues()
    a5, b5 = CFR(100).get_poles_and_residues()

    energy = np.linspace(-5.7, 5.7, 3000)

    temp = 300
    fd0 = fd(energy, 0, temp)

    kb = 8.61733e-5  # Boltzmann constant in eV
    energy = energy / (kb * temp)

    ans1 = approximant(energy, a1, b1)
    ans2 = approximant(energy, a2, b2)
    ans3 = approximant(energy, a3, b3)
    ans4 = approximant(energy, a4, b4)
    ans5 = approximant(energy, a5, b5)

    return fd0, ans1, ans2, ans3, ans4, ans5


if __name__ == '__main__':
    fd0, ans1, ans2, ans3, ans4, ans5 = test_approximation()

    ans6 = test_itegrate()
    ans7 = test_itegrate1()

    gf_array = test_gf(np.linspace(-12, 12, 150) + 1e-2 * 1j)

    fig, ax = plt.subplots(3)
    ax[0].plot(fd0)
    ax[0].plot(ans1)
    ax[0].plot(ans2)
    ax[0].plot(ans3)
    ax[0].plot(ans4)
    ax[0].plot(ans5)

    ax[1].plot(gf_array)

    ax[2].plot(ans6)
    ax[2].plot(ans7)
    plt.show()
