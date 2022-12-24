#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES='2'

source activate 

pred_path="/nvme/suyanzhou/projects/MM/Exp/Polyp/FPN_PVT/lightning_logs/version_39/result_map"
result_path="/nvme/suyanzhou/projects/MM/Exp/Polyp/FPN_PVT/lightning_logs/version_39/result_map"
gt_path="/nvme/suyanzhou/dataset/dataset_2d/Polyp/TestDataset"
# eval_datasets='CVC-300','CVC-ClinicDB','Kvasir','CVC-ColonDB','ETIS-LaribPolypDB'
eval_datasets='Kvasir'

python -W ignore eval_modify.py \
    --pred_path $pred_path \
    --result_path $result_path \
    --gt_path $gt_path \
    --datasets=${eval_datasets}