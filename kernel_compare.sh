#!/bin/bash
powers_of_2=(4)
list_of_bandlimit=(8)
spectral_dirs=(extended)

# zipped iterator on powers_of_2 and list_of_bandlimit
for indx in "${!powers_of_2[@]}"; do
    nside=${powers_of_2[$indx]}
    bandlimit=${list_of_bandlimit[$indx]}
    echo "*****************************************"
    for spectral_dir in "${spectral_dirs[@]}"; do

        echo "############################################################"
        
        echo "Running test for spectral_dir=$spectral_dir, nside=$nside"

        ./build/_s2fft --nside $nside --type $spectral_dir -L $bandlimit --print &> _logs/cuda_output.log
        python _scripts/kernel_test.py --nside $nside --type $spectral_dir -L $bandlimit --print &> _logs/python_output.log

    done
done