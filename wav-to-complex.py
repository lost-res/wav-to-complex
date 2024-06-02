# Convert .wav IQ file to complex file.
# Usage -- python3 ./wav-to-complex.py <infile> <outFile>
# 
#

import librosa
import numpy as np
import torchaudio
import sys
from scipy.io.wavfile import read as read_wav


# Get input & output file from passed arguments
inFile = sys.argv[1]
outFile = sys.argv[2]

sampleRate, data = read_wav(inFile)
metadata = torchaudio.info(inFile)
print(metadata)

# Interleave I & Q from 2 channel
iqData = np.vstack( ( data[0][0], data[0][1] ) ).reshape( (-1,), order='F' )

# Write interleaved I & Q data to outFile in complex64 format
iqData.tofile( outFile )
