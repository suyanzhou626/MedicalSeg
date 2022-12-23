#!/bin/bash
source activate 

python main_v2.py  --config ./config/config-CVC-ClinicDB.yaml --model.model_name 'FPN_PVT'  --trainer.devices ['6']