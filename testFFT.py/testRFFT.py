from scipy.fftpack import rfft,irfft

import numpy as np

x=np.random.random((1,6))

print(x)
print(rfft(x))
print(irfft(rfft(x)))

