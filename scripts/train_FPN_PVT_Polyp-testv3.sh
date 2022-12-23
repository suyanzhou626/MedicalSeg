#!/bin/bash
source activate 

python main_v3.py  --config ./config/config-Polyp-FPN_PVT.yaml --trainer.devices ['7']  --trainer.max_epochs 100