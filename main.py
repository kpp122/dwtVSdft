import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_data(datafile):
    data = pd.read_csv(datafile, sep=",", header=0)
    data.drop(columns=["Year", "m", "d", "Time", "Time zone"], inplace=True)
    data.replace('-', 0, inplace=True)
    data = data.astype("float32")

    log_len = int(np.floor(np.log2(len(data))))  # for simplicity, we use n that is a power of 2

    return data[:2 ** log_len]


def calc_weights(data):
    tmp = data.copy()
    weights = []
    half = int(len(tmp) / 2)
    sums = [0] * half
    j = 1

    while True:
        for i in range(half):
            _sum = tmp[i * 2] + tmp[i * 2 + 1]
            weights.append((tmp[i * 2] - tmp[i * 2 + 1]) / (2 ** j))
            sums[i] = _sum

        if half == 1:
            weights.append(sum(data) / len(data))
            return np.array(weights)

        half = int(half / 2)
        tmp = sums
        j += 1


def wavelets(n):
    matrix = np.zeros((n, n))
    i = 0
    k = 2

    for row in matrix:
        start = i * k
        stop = i * k + int(k / 2)

        row[start: stop] = 1
        row[stop: stop + int(k / 2)] = -1

        if stop + int(k / 2) == len(row):
            k = 2 * k
            i = 0
        else:
            i += 1

    return matrix


def approximation(weights, wavelets, m):
    E = energy(weights, wavelets)
    to_keep = np.argsort(E)[-m:]

    retained_energy = round(sum(np.array(E)[to_keep]) / sum(E) * 100, 2)

    w_prime = np.array(weights, copy=True)

    for i in range(len(w_prime)):
        if i not in to_keep:
            w_prime[i] = 0

    approx = np.dot(np.transpose(w_prime), wavelets)
    return approx, retained_energy


def energy(weights, wavelets):
    energy = []

    for i in range(len(weights)):
        energy.append((weights[i] ** 2) * np.sum(np.not_equal(wavelets[i], 0)))

    return energy


def plot_results(title, original, dwt, dft, energy, energy_dft):

    fig, ax = plt.subplots(3, sharey=True, sharex="col")
    plt.suptitle(title)
    ax[0].plot(original)
    ax[0].set_title("original")

    ax[1].plot(dwt)
    ax[1].set_title("dwt, retained energy %s" % energy + " %")

    ax[2].plot(dft)
    ax[2].set_title("dft, retained energy %s" % energy_dft + " %")
    plt.tight_layout()

    plt.show()


if __name__ == '__main__':

    path = "weather2.csv"
    data = read_data(path)
    fraction = 1/4

    for attribute in data.columns:

        dat = list(data[attribute])
        size = int(len(dat) * fraction)

        weights = calc_weights(dat)
        wavelet_matrix = wavelets(len(dat))
        approx, retained_energy = approximation(weights, wavelet_matrix, size)

        dft = np.fft.rfft(dat)

        retained_energy_dft = sum(abs(dft[:int(len(dft) * fraction)])**2) / sum(abs(dft**2))
        retained_energy_dft = round(retained_energy_dft * 100, 2)
        dft[int(len(dft) * fraction):] = 0

        dft = np.fft.irfft(dft)
        plot_results(attribute, dat, approx, dft, retained_energy, retained_energy_dft)


