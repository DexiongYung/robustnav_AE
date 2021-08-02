import os

models = ['VAE', 'AE', 'LUSR', 'DDVAE']
script = "python main.py -o=storage/robothor-objectnav-rgb-custom-ddppo -b=projects/robustnav_baselines/experiments/robustnav_train objectnav_robothor_vanilla_rgb_custom_ddppo -s=12345 -et=custom -em={m}_64 -emb=domain_adaptation/models/M64 -las=64 -sckpt=./domain_adaptation/checkpoints/{m}/weather/best_model.pt"

for m in models:
    print(f'Running Model: {m}')
    os.system(script.format(m = m))