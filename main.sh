#!/bin/bash

FILE=./pysot/experiments/siammask_r50_l3/model.pth
if [ -f "$FILE" ]; then
    echo "$FILE exists."
else 
    echo "$FILE does not exist. Starting download..."
    gdown https://drive.google.com/uc?id=1dQoI2o5Bzfn_IhNJNgcX4OE79BIHwr8s 
    mv model.pth ./pysot/experiments/siammask_r50_l3
fi
export PYTHONPATH=./pysot:$PYTHONPATH
python label.py \
        --label $1 \
        --output $2 \
        --config ./pysot/experiments/siammask_r50_l3/config.yaml \
        --snapshot ./pysot/experiments/siammask_r50_l3/model.pth \
        --video $3 \
        --decimate $4
