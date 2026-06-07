accelerate launch pretrain_vae.py \
        --mask_ratio 0.3 \
        --root_dir "." \
        --config_name "dpf_config.json" \
        --output_dir "CrysLDNet_pretrain_vae"
