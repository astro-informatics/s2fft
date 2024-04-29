#!/bin/bash
powers_of_2=(256)
normalizations=(bwd)
directions=("fwd" "bwd")

for nside in "${powers_of_2[@]}"; do
    echo "*****************************************"
    for norm in "${normalizations[@]}"; do
        for dir in "${directions[@]}"; do

            echo "############################################################"
            
            echo "Running test for nside=$nside, norm=$norm, dir=$dir with shift"

            ./build/_s2fft --nside $nside --norm $norm --ffttype $dir --shift --print > _logs/cuda_output.log
            python _scripts/fft_test.py --nside $nside --norm $norm --ffttype $dir --shift --print > _logs/python_output.log
            python _scripts/comparer.py -f _logs/cuda_output.log _logs/python_output.log

            #echo "Running test for nside=$nside, norm=$norm, dir=$dir without shift"
#
            #./build/_s2fft --nside $nside --norm $norm --ffttype $dir --print > _logs/cuda_output_second.log
            #python _scripts/fft_test.py --nside $nside --norm $norm --ffttype $dir --print > _logs/python_output_second.log
            #python _scripts/comparer.py -f _logs/cuda_output_second.log _logs/python_output_second.log

        done
    done
done