#!/bin/bash

powers_of_2=(4 8 32 64 128 256 512)
powers_of_2=(128)

for nside in "${powers_of_2[@]}"; do
    echo "*****************************************"
    echo "Running tests for nside=$nside"
    echo "Norm  Ortho  Shift"
    ./build/_s2fft --nside $nside --norm 'ortho' --shift --test
    echo "Norm  Ortho NoShift"
    ./build/_s2fft --nside $nside --norm 'ortho' --test
    echo "Norm  Bwd    Shift"
    ./build/_s2fft --nside $nside --norm 'bwd' --shift --test
    echo "Norm  Bwd    NoShift"
    ./build/_s2fft --nside $nside --norm 'bwd' --test
    echo "Norm  Fwd    Shift"
    ./build/_s2fft --nside $nside --norm 'fwd' --shift --test
    echo "Norm  Fwd    NoShift"
    ./build/_s2fft --nside $nside --norm 'fwd' --test
done
