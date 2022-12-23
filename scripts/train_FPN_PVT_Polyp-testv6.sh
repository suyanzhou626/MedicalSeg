#!/bin/bash
source activate 

python main_v6.py  --config ./config/config-Polyp-FPN_PVT.yaml --trainer.devices ['6']  --trainer.max_epochs 1  --trainer.gradient_clip_val 0.5