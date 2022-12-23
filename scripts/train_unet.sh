#!/bin/bash
source activate 

python main.py fit --seed 1 --config config.yaml
python main.py test --seed 1 --config config.yaml
