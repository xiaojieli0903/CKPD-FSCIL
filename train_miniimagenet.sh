GPUS=8
work_dir=work_dirs/ckpd_fscil/miniimagenet_vit_ckpd_fscil_1gpu_freeze11_lr0.005
bash tools/dist_train.sh configs/miniimagenet/vit_etf_bs512_500e_miniimagenet_ckpd_fscil.py $GPUS --work-dir ${work_dir} --seed 1 --deterministic
GPUS=1
bash tools/run_fscil.sh configs/miniimagenet/vit_etf_bs512_500e_miniimagenet_eval_ckpd_fscil_1e-2_base0.1.py ${work_dir} ${work_dir}/best.pth $GPUS --seed 0 --deterministic