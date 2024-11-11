#!/bin/bash
pwd
echo $CONDA_DEFAULT_ENV

## build vocabulary files - no overlap, 7-mer, length 3 interactor
python3 ../../swing_roberta/build_vocabulary_files.py \
        --data_dir ../npflip_nature_mut_wt_merged_groups.csv \
        --out_dir ./ \
        --k 7 \
        --sub_size 7 \
        --l 1 \
        --freq -1 \
        --type 'MUTINT'
                                  
