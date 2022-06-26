#!bin/bash
which nvcc
echo "***************"
nvcc --version
echo "***************"
cd /usr/local/cuda-11.1/samples/1_Utilities/deviceQuery/
make
./deviceQuery
