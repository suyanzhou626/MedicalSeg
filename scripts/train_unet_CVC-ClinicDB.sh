#!/bin/bash
source activate 

python main_v1.py  --config ./config/config-CVC-ClinicDB.yaml --trainer.devices ['7']
