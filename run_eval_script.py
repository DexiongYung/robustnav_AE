import os

model_base = 'domain_adaptation/models/M64'
config = [('VAE_64', './domain_adaptation/checkpoints/VAE/weather/best_model.pt', './storage/vae/checkpoints/Pointnav-RoboTHOR-Vanilla-RGB-ResNet-DDPPO/custom/2021-08-09_06-17-30/exp_Pointnav-RoboTHOR-Vanilla-RGB-ResNet-DDPPO_custom__stage_00__steps_000030005760.pt', '2021-08-09_06-17-30', 'VAE_64'),
('VAE_64', './domain_adaptation/checkpoints/DVAE/weather/best_model.pt', './storage/dvae/checkpoints/Pointnav-RoboTHOR-Vanilla-RGB-ResNet-DDPPO/custom/2021-08-09_21-45-36/exp_Pointnav-RoboTHOR-Vanilla-RGB-ResNet-DDPPO_custom__stage_00__steps_000030005760.pt', '2021-08-09_21-45-36', 'DVAE_64'),
('VAE_64', './domain_adaptation/checkpoints/DDVAE/weather/best_model.pt', './storage/ddvae/checkpoints/Pointnav-RoboTHOR-Vanilla-RGB-ResNet-DDPPO/custom/2021-08-06_18-15-19/exp_Pointnav-RoboTHOR-Vanilla-RGB-ResNet-DDPPO_custom__stage_00__steps_000030005760.pt', '2021-08-06_18-15-19', 'DDVAE_64')]
corruption_list = ['Defocus_Blur', 'Motion_Blur', 'Spatter', 'Low_Lighting', 'Speckle_Noise']
latent_size = 64
severity_list = list(range(1,6))
task = 'pointnav'

for model, model_ckpt, rn_ckpt, date, name in config:
    for corruption in corruption_list:
        for severity in severity_list:
            tensorboard_name = f'{name}_{corruption}{severity}'
            command = f'python main.py -o=storage/{tensorboard_name} -b=projects/robustnav_baselines/experiments/robustnav_eval {task}_robothor_vanilla_rgb_custom_ddppo -s=12345 -et=custom -em={model} -emb={model_base} -las={latent_size} -sckpt={model_ckpt} -c={rn_ckpt} -t={date}'
            os.system(command)