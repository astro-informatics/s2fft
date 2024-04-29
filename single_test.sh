#!/bin/bash

nside=256
norm='bwd'
dir='bwd'

./build/_s2fft --nside $nside --norm $norm --ffttype $dir --shift --print > _logs/cuda_output.log
python _scripts/fft_test.py --nside $nside --norm $norm --ffttype $dir --shift --print > _logs/python_output.log
python _scripts/comparer.py -f _logs/cuda_output.log _logs/python_output.log


dir='fwd'


./build/_s2fft --nside $nside --norm $norm --ffttype $dir --shift --print > _logs/cuda_output_second.log
python _scripts/comparer.py -f _logs/cuda_output.log _logs/cuda_output_second.log

python _scripts/fft_test.py --nside $nside --norm $norm --ffttype $dir --shift --print > _logs/python_output_second.log
python _scripts/comparer.py -f _logs/python_output_second.log _logs/python_output.log

python _scripts/comparer.py -f _logs/cuda_output_second.log _logs/python_output_second.log