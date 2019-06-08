import numpy as np
import struct
import pandas

def mean_(data):
    avg = np.zeros((data.shape[0], data.shape[2]))
    cnt = np.zeros((data.shape[0], data.shape[2]))
    for i in range(data.shape[0]):
        for j in range(data.shape[2]):
            for k in range(25):
                # if data[i, k, j] == 9999: print("!!")
                if data[i, k, j] == 0 or data[i, k, j] == 9999: continue
                avg[i, j] += data[i, k, j]
                cnt[i, j] += 1
            avg[i, j] /= cnt[i, j]
    return avg

def normalize(data):
    data = data.copy()
    avg = np.zeros((data.shape[0], data.shape[2]))
    max_ = np.zeros((data.shape[0], data.shape[2]))
    max_.fill(-9999)
    min_ = np.zeros((data.shape[0], data.shape[2]))
    min_.fill(9999)
    cnt = np.zeros((data.shape[0], data.shape[2]))
    for i in range(data.shape[0]):
        for j in range(data.shape[2]):
            for k in range(25):
                if data[i, k, j] == 0 or data[i, k, j] == 9999: continue
                avg[i, j] += data[i, k, j]
                max_[i, j] = max(max_[i, j], data[i, k, j])
                min_[i, j] = min(min_[i, j], data[i, k, j])
                cnt[i, j] += 1
            avg[i, j] /= cnt[i, j]
            for k in range(25):
                if data[i, k, j] == 0 or data[i, k, j] == 9999:
                    data[i, k, j] = 0
                else:
                    data[i, k, j] -= avg[i, j]
            for k in range(25):
                data[i, k, j] /= (max_[i, j] - min_[i, j])

    return data

def fetch(filename):
    depth = []

    with open(filename, 'rb') as f:
        while True:
            x = f.read(2)
            if x == b'':
                break
            v = struct.unpack('H', x)
            depth.append(v)

    depth = np.reshape(depth, (H, W))
    depth = depth / np.max(depth)
    return depth


def process(file):
    data = pandas.read_csv(file, header=None)
    data = np.array(data)
    data = data.reshape((-1, 150, 5))
    index = 0
    while np.sum(data[0, index: index + 25, 0]) == 0:
        index += 25
    return data[:, index: index + 25, :3]

