
import numpy as np
from numpy.testing import assert_allclose
from numpy.fft import fft , ifft , fftshift , ifftshift
from math import ceil


nside = 8
offsets = { "upper" : [] , "lower" : [] , "sizes" : []} 
#total_pxiels = nside**2 * 12
#a = np.arange(total_pxiels)

start_index = 0
end_index = (12 * nside * nside)

for i in range(nside - 1):
    nphi = 4 * (i + 1)

    offsets["upper"].append(start_index)
    offsets["lower"].append(end_index - nphi)
    offsets["sizes"].append(nphi)
    start_index += nphi
    end_index -= nphi



def find_offset(offsets , index):
    
    for i in range(len(offsets["upper"])):
        if index <= offsets["upper"][i]:
            return offsets["upper"][(i - 1)] , offsets["sizes"][(i - 1)] , i , "upper"
        elif index >= offsets["lower"][i]:
            return offsets["lower"][i] , offsets["sizes"][i] , i , "lower"
    return -1 , -1 , "none"

print(find_offset(offsets , 112))