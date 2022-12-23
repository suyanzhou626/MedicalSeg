#!/bin/bash
source activate 

python main_v2.py  --config ./config/config-Polyp-FPN_PVT.yaml --trainer.devices ['6']