#!/bin/bash
source activate 

python main_v1.py  --config ./config/config-CVC-ClinicDB.yaml --model.model_name 'NestedUNet'  --trainer.devices ['6']