
import numpy as np
from numpy.fft import fft,ifft,fftshift,ifftshift



a = np.arange(24420 , 24420 + 444)

for i,elem in enumerate((fft(a))):
    print(f"[{i+24420}] {elem.real} + {elem.imag}j")