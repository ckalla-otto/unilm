#!/bin/bash
python run_class_finetuning.py --model beit_base_patch16_224 --data_path "/generated/cache/basic_color_samples" \
                --nb_classes 43 --data_set "tfrecord" --disable_eval_during_finetuning \
                --finetune "https://unilm.blob.core.windows.net/beit/beit_base_patch16_224_pt22k_ft22k.pth" \
                --output_dir "output_dir" --batch_size 64 --lr 2e-3 \
                --warmup_epochs 5 --epochs 90 --layer_decay 0.75 --drop_path 0.2 \
                --weight_decay 0.05 --layer_scale_init_value 1e-5 --clip_grad 1.0 \
                --device cuda \
                --eval