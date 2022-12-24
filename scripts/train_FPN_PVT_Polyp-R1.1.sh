#!/bin/bash
source activate 

python main_R1.1.py  --config ./config/config-Polyp-FPN_PVT.yaml --trainer.devices ['7']  --trainer.gradient_clip_val 0.5 --trainer.max_epochs 120