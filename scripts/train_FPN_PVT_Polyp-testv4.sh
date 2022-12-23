#!/bin/bash
source activate 

python main_v4.py  --config ./config/config-Polyp-FPN_PVT.yaml --trainer.devices ['5']  --trainer.max_epochs 100  --trainer.gradient_clip_val 0.5