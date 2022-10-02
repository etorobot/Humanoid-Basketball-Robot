#refer to the file and class for each env
from gym.envs.registration import register

register(id= 'KaleidoBed-v0',   entry_point= 'muj_envs_folder.list_of_muj_envs:KHumanoidStandupClass'
        ,max_episode_steps=500 
        )
register(id= 'KHumanoid-v0',    entry_point= 'muj_envs_folder.list_of_muj_envs:KHumanoidClass')
register(id= 'KaleidoKHI-v0',   entry_point= 'muj_envs_folder.list_of_muj_envs:khi_class'
        ,max_episode_steps=2000
        # ,kwargs={'config': 'randomizer/config/KaleidoKHIRandom/random.json'}
        )
register(id= 'hold-v0',        entry_point= 'muj_envs_folder.list_of_muj_envs:hold_class')
register(id= 'blanK-v3',        entry_point= 'muj_envs_folder.list_of_muj_envs:blank')
