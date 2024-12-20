GPUS=1
work_dir=work_dirs/ckpd_fscil/cub_vit_ckpd_fscil_1gpu_freeze11_lr0.005
bash tools/dist_train.sh configs/cub/vit_etf_bs512_80e_cub_ckpd_fscil.py $GPUS --work-dir ${work_dir} --seed 1 --deterministic
bash tools/run_fscil.sh configs/cub/vit_etf_bs512_80e_cub_eval_ckpd_fscil_bigaug_1e-2_base0.1.py ${work_dir} ${work_dir}/best.pth $GPUS --seed 0 --deterministic