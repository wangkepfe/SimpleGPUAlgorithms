import numpy as np
import matplotlib.pyplot as plt
import halton

def concentric_disk_sampling(u):
    uOffset = u * 2.0 - 1.0
    if abs(uOffset[0]) < 1e-10 and abs(uOffset[1]) < 1e-10:
        return np.array([[0.0,0.0]])
    r = 0.0
    theta = 0.0
    if abs(uOffset[0]) > abs(uOffset[1]):
        r = uOffset[0]
        theta = np.pi / 4 * (uOffset[1] / uOffset[0])
    else:
        r = uOffset[1]
        theta = np.pi / 2 - np.pi / 4 * (uOffset[0] / uOffset[1])
    return r * np.array([[np.cos(theta), np.sin(theta)]])

def get_concentric_disk_samples(squareLength, skipping = 0):

    pointsInSquare = []

    for i in range(squareLength):
        for j in range(squareLength):
            if skipping > 0 and (i + j * squareLength) % (skipping + 1) == 0:
                continue
            pointsInSquare.append([float(i),float(j)])

    pointsInSquare = np.asarray(pointsInSquare)
    pointsInSquare += 0.5
    pointsInSquare /= float(squareLength)

    resultList = np.empty((0, 2), float)
    for point in pointsInSquare:
        result = concentric_disk_sampling(point)
        result *= 5.0
        resultList = np.append(resultList, result, axis=0)

    return resultList

def get_halton_concentric_samples(size):
    seq = halton.halton_sequence2(size,[2,3])
    seq = np.transpose(seq)
    resultList = np.empty((0, 2), float)
    for pt in seq:
        result = concentric_disk_sampling(pt)
        resultList = np.append(resultList, result, axis=0)
    return resultList

