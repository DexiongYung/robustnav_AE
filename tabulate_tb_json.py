import os
import json
import numpy as np
from os import listdir
from os.path import isfile, join


model_list = ['NoPretrain'] # ['VAE', 'DDVAE', 'DVAE']
corruption_list = ['Defocus_Blur'] #, 'Motion_Blur', 'Spatter', 'Low_Lighting', 'Speckle_Noise', 'cam_crack', 'fov']
levels = [1, 3, 5]
path = './storage'

for model in model_list:
    model_name_len = len(model)
    model_dist, model_ep, model_reward, model_spl, model_success = list(), list(), list(), list(), list()
    for corruption in corruption_list:
        for l in levels:
            json_files = list()
            folders = list()
            dir_list = [x for x in os.listdir(path)]
            for dir in dir_list:
                has_level = str(l) in dir
                
                if model == dir[:model_name_len] and corruption in dir and has_level:
                    folders.append(join(path,dir))

            for folder in folders:
                for p, dir_names, f in list(os.walk(folder)):
                    if len(f) > 0 and '.json' in f[0]:
                        json_files.append(p + '/' + f[0])

            dist, ep_len, reward, spl, success = list(), list(), list(), list(), list()

            for f in json_files:
                f_split = f.split('/')
                for j in json_files:
                    j_split = j.split('/')

                    if j_split[0] != f_split[0]:
                        raise ValueError(j_split[0] + ' does not equal ' + f_split[0])
                    elif j_split[1] != f_split[1]:
                        raise ValueError(j_split[1] + ' does not equal ' + f_split[1])
                    elif j_split[2][:-1] != f_split[2][:-1]:
                        raise ValueError(j_split[2][:-1] + ' does not equal ' + f_split[2][:-1])


            for j_path in json_files:
                f = open(j_path)
                curr_json = json.load(f)[0]
                dist.append(curr_json['dist_to_target'])
                ep_len.append(curr_json['ep_length'])
                reward.append(curr_json['reward'])
                spl.append(curr_json['spl'])
                success.append(curr_json['success'])
            
            avg_dist = round(np.mean(dist), 3)
            avg_ep = round(np.mean(ep_len),3)
            avg_reward = round(np.mean(reward),3)
            avg_spl = round(np.mean(spl),3)
            avg_success = round(np.mean(success),3)

            sd_dist = round(np.std(dist), 3)
            sd_ep = round(np.std(ep_len), 3)
            sd_reward = round(np.std(reward), 3)
            sd_spl = round(np.std(spl), 3)
            sd_success = round(np.std(success), 3)

            # model_dist.append(avg_dist)
            # model_ep.append(avg_ep)
            # model_reward.append(avg_reward)
            # model_spl.append(avg_spl)
            # model_success.append(avg_success)
            
            print(f'Model: {model}, Corruption: {corruption}, Level: {l}')
            print(f'Dist to target: {avg_dist}+-{sd_dist}')
            print(f'Ep Len: {avg_ep}+-{sd_ep}')
            print(f'Reward: {avg_reward}+-{sd_reward}')
            print(f'SPL: {avg_spl}+-{sd_spl}')
            print(f'Succeed: {avg_success}+-{sd_success}')
            print(f'{avg_dist}$\pm${sd_dist}&{avg_ep}$\pm${sd_ep}&{avg_reward}$\pm${sd_reward}&{avg_spl}$\pm${sd_spl}&{avg_success}$\pm${sd_success}')
    
    # all_dist = round(np.mean(model_dist), 3)
    # all_ep = round(np.mean(model_ep), 3)
    # all_reward = round(np.mean(model_reward), 3)
    # all_spl = round(np.mean(model_spl), 3)
    # all_success = round(np.mean(model_success), 3)

    # print(f'Model: {model}, dist: {all_dist}, ep: {all_ep}, reward: {all_reward}, spl: {all_spl}, succ: {all_success}')
    # print(f'{all_dist}&{all_ep}&{all_reward}&{all_spl}&{all_success}')