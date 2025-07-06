# baseline run
python run_expid.py --config config/DIN_microlens_mmctr_tuner_config_01 --expid DIN_MicroLens_1M_x1_002_1fa8d93d --gpu 0

# ours run (batch_size=256)
python run_expid.py --config config/Transformer_DCN_microlens_mmctr_tuner_config_01 --expid Transformer_DCN_MicroLens_1M_x1_001_820c435c --gpu 4

# ours run (batch_size=128)
python run_expid.py --config config/Transformer_DCN_microlens_mmctr_tuner_config_01 --expid Transformer_DCN_MicroLens_1M_x1_001_323m436f --gpu 0


