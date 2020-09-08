import numpy as np


def manhattanDistance(p1, p2, d=2):
    assert (len(p1) == len(p2))
    allDistances = 0
    for i in range(d):
        allDistances += abs(p1[i] - p2[i])
    return allDistances


def printBigArray(big, yes='1', no='0'):
    print()
    for line in big:
        for char in line:
            if char:
                print(yes, end=" ")
            else:
                print(no, end=" ")
        print()


def removeDupsFromPointList(ptList):
    newList = ptList[:]
    return tuple(set(newList))

def arrayToWhiteGreyscalePixel(array, pixels):
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i, j]:
                pixels[i, j] = 255

def padVoxelArray(voxels):
    """
    Adds one voxel in each direction (negative and positive)
    and returns the new bounding box size
    """
    a, b, c = voxels.shape
    vol = np.zeros((a + 2, b + 2, c + 2), dtype=bool)
    vol[1:a+1,1:b+1,1:c+1] = voxels
    return vol, (b, c, a)
