#!/bin/bash

#BSUB -q GPU
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o stdout_%J.out
#BSUB -e stdout_%J.err

BASE_DIR=/F800/wangxingpei/repositories/bert
MODEL_DIR=${BASE_DIR}/models/chinese_L-12_H-768_A-12

python run_classifier.py \
  --task_name=text \
  --do_train=true \
#  --do_eval=true \
  --do_predict=true \
  --data_dir=${BASE_DIR}/dataset/simplifyweibo_4_moods \
  --vocab_file=${MODEL_DIR}/vocab.txt \
  --bert_config_file=${MODEL_DIR}/bert_config.json \
  --init_checkpoint=${MODEL_DIR}/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=16 \
  --learning_rate=2e-5 \
  --num_train_epochs=20 \
  --output_dir=${BASE_DIR}
