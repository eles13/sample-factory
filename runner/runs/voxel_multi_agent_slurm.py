from runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('env', ['voxel_env_TowerBuilding', 'voxel_env_ObstaclesEasy', 'voxel_env_ObstaclesHard', 'voxel_env_Collect', 'voxel_env_Sokoban', 'voxel_env_HexMemory', 'voxel_env_HexExplore', 'voxel_env_Rearrange']),
])

_cli = 'python -m algorithms.appo.train_appo --train_for_seconds=360000000 --train_for_env_steps=2000000000 --algo=APPO --gamma=0.997 --use_rnn=True --rnn_num_layers=2 --num_workers=14 --num_envs_per_worker=2 --ppo_epochs=1 --rollout=32 --recurrence=32 --batch_size=2048 --actor_worker_gpus 0 --num_policies=1 --with_pbt=False --max_grad_norm=0.0 --exploration_loss=symmetric_kl --exploration_loss_coeff=0.001 --voxel_num_simulation_threads=1 --voxel_use_vulkan=True --policy_workers_per_policy=2 --learner_main_loop_num_cores=1'

_experiment_2agents = Experiment(
    'voxel_env_2ag',
    _cli + ' --voxel_num_envs_per_instance=18 --voxel_num_agents_per_env=2',
    _params.generate_params(randomize=False),
)

_experiment_4agents = Experiment(
    'voxel_env_4ag',
    _cli + ' --voxel_num_envs_per_instance=9 --voxel_num_agents_per_env=4',
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('voxel_env_v114_mutli_agent_v53', experiments=[_experiment_2agents, _experiment_4agents])
