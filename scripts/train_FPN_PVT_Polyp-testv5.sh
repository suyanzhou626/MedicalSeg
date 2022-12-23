#!/bin/bash
source activate 

python main_v5.py  --config ./config/config-Polyp-FPN_PVT.yaml --trainer.devices ['6']  --trainer.max_epochs 100  --trainer.gradient_clip_val 0.5 --data.num_workers 0 --data.batch_size 2