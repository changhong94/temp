#!/bin/bash

nvcc -arch=sm_86 -g -cudart shared -o test_ckp matmul.cu
