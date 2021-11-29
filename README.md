
# Influence-Based Reinforcement Learning for Intrinsically-Motivated Agents
## Installation instructions

Install Python packages
```shell
# require Anaconda 3 or Miniconda 3
bash install_dependecies.sh
```

Set up StarCraft II (2.4.10) and SMAC:
```shell
bash install_sc2.sh
```

This will download SC2.4.10 into the 3rdparty folder and copy the maps necessary to run over.

Set up Google Football:
```shell
bash install_gfootball.sh
```

## Command Line Tool

**Run an experiment**

```shell
# For SMAC (QMIX)
python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=corridor
```

```shell
# For SMAC (ALITA)
python3 src/main.py --config=alita --env-config=sc2 with env_args.map_name=corridor
```

```shell
# For Difficulty-Enhanced Predator-Prey
python3 src/main.py --config=qmix_predator_prey --env-config=stag_hunt with env_args.map_name=stag_hunt
```

```shell
# For Communication tasks
python3 src/main.py --config=qmix_att --env-config=sc2 with env_args.map_name=1o_10b_vs_1r
```



The config files act as defaults for an algorithm or environment. 

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`

**Run n parallel experiments**

```shell
# bash run.sh config_name env_config_name map_name_list (arg_list threads_num gpu_list experinments_num)
bash run.sh qmix sc2 6h_vs_8z epsilon_anneal_time=500000,td_lambda=0.3 2 0 5
```

`xxx_list` is separated by `,`.

All results will be stored in the `Results` folder and named with `map_name`.

**Kill all training processes**

```shell
# all python and game processes of current user will quit.
bash clean.sh
```

# Citation
```
@article{fayad2021influence,
  title={Influence-Based Reinforcement Learning for Intrinsically-Motivated Agents},
  author={Fayad, Ammar and Ibrahim, Majd},
  journal={arXiv preprint arXiv:2108.12581},
  year={2021}
}

```

