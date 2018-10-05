#!/bin/bash
#PBS -l select=1:ncpus=272 -lplace=excl

export PYTHONPATH=$PYTHONPATH:/export/software/tensorflow-1.3.0-rc2/python_modules/

source /export/software/GCC/source_swtool.sh 
source /export/software/GCC/source_swtool_63.sh

source ~/virtualenv/keras/bin/activate

export OMP_NUM_THREADS=136
export KMP_AFFINITY=granularity=fine,compact,1,0;

cd ~/vnet_keras
stdbuf -o 0 python predict_training_data.py 2>&1 | tee ~/vnet_keras/results/test_impl_3.log