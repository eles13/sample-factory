import copy
import json
import math
import numbers
import os
import random
import time
from enum import Enum
from queue import Full
from os.path import join

import numpy as np

from sample_factory.algorithms.appo.appo_utils import TaskType, iterate_recursively
from sample_factory.algorithms.utils.algo_utils import EPS
from sample_factory.utils.utils import log, experiment_dir


def perturb_float(x, perturb_amount=1.2):
    # mutation direction
    new_value = x / perturb_amount if random.random() < 0.5 else x * perturb_amount
    return new_value


def perturb_vtrace(x, cfg):
    return perturb_float(x, perturb_amount=1.005)


def perturb_exponential_decay(x, cfg):
    perturbed = perturb_float(1.0 - x)
    new_value = 1.0 - perturbed
    new_value = max(EPS, new_value)
    return new_value


def perturb_batch_size(x, cfg):
    new_value = perturb_float(x, perturb_amount=1.2)
    initial_batch_size = cfg.batch_size
    max_batch_size = initial_batch_size * 1.5
    min_batch_size = cfg.rollout

    new_value = min(new_value, max_batch_size)

    # round down to whole number of rollouts
    new_value = (int(new_value) // cfg.rollout) * cfg.rollout

    new_value = max(new_value, min_batch_size)
    return new_value


class PbtTask(Enum):
    SAVE_MODEL, LOAD_MODEL, UPDATE_CFG, UPDATE_REWARD_SCHEME = range(4)


HYPERPARAMS_TO_TUNE = {
    'learning_rate', 'exploration_loss_coeff', 'value_loss_coeff', 'max_grad_norm', 'ppo_clip_ratio', 'ppo_clip_value',
}

# if not specified then tune all rewards
REWARD_CATEGORIES_TO_TUNE = {
    'doom_': ['delta', 'selected_weapon'],
}

# HYPERPARAMS_TO_TUNE_EXTENDED = {
#     'learning_rate', 'exploration_loss_coeff', 'value_loss_coeff', 'adam_beta1', 'max_grad_norm',
#     'ppo_clip_ratio', 'ppo_clip_value', 'vtrace_rho', 'vtrace_c',
# }

SPECIAL_PERTURBATION = dict(
    gamma=perturb_exponential_decay,
    adam_beta1=perturb_exponential_decay,
    vtrace_rho=perturb_vtrace,
    vtrace_c=perturb_vtrace,
    batch_size=perturb_batch_size,
)


def policy_cfg_file(cfg, policy_id):
    return join(experiment_dir(cfg=cfg), f'policy_{policy_id:02d}_cfg.json')


def policy_reward_shaping_file(cfg, policy_id):
    return join(experiment_dir(cfg=cfg), f'policy_{policy_id:02d}_reward_shaping.json')


class PopulationBasedTraining:
    def __init__(self, cfg, default_reward_shaping, summary_writers):
        self.cfg = cfg

        if cfg.pbt_optimize_batch_size and 'batch_size' not in HYPERPARAMS_TO_TUNE:
            HYPERPARAMS_TO_TUNE.add('batch_size')

        self.last_update = [0] * self.cfg.num_policies

        self.policy_cfg = [dict() for _ in range(self.cfg.num_policies)]
        self.policy_reward_shaping = [dict() for _ in range(self.cfg.num_policies)]

        self.default_reward_shaping = default_reward_shaping

        self.summary_writers = summary_writers
        self.last_pbt_summaries = 0

        self.learner_workers = self.actor_workers = None

        self.reward_categories_to_tune = []
        for env_prefix, categories in REWARD_CATEGORIES_TO_TUNE.items():
            if cfg.env.startswith(env_prefix):
                self.reward_categories_to_tune = categories

    def init(self, learner_workers, actor_workers):
        self.learner_workers = learner_workers
        self.actor_workers = actor_workers

        for policy_id in range(self.cfg.num_policies):
            # save the policy-specific configs if they don't exist, or else load them from files
            policy_cfg_filename = policy_cfg_file(self.cfg, policy_id)
            if os.path.exists(policy_cfg_filename):
                with open(policy_cfg_filename, 'r') as json_file:
                    log.debug('Loading initial policy %d configuration from file %s', policy_id, policy_cfg_filename)
                    json_params = json.load(json_file)
                    self.policy_cfg[policy_id] = json_params
            else:
                self.policy_cfg[policy_id] = dict()
                for param_name in HYPERPARAMS_TO_TUNE:
                    self.policy_cfg[policy_id][param_name] = self.cfg[param_name]

                if policy_id > 0:  # keep one policy with default settings in the beginning
                    log.debug('Initial cfg mutation for policy %d', policy_id)
                    self.policy_cfg[policy_id] = self._perturb_cfg(self.policy_cfg[policy_id])

        for policy_id in range(self.cfg.num_policies):
            # save the policy-specific reward shaping if it doesn't exist, or else load from file
            policy_reward_shaping_filename = policy_reward_shaping_file(self.cfg, policy_id)

            if os.path.exists(policy_reward_shaping_filename):
                with open(policy_reward_shaping_filename, 'r') as json_file:
                    log.debug(
                        'Loading policy %d reward shaping from file %s', policy_id, policy_reward_shaping_filename,
                    )
                    json_params = json.load(json_file)
                    self.policy_reward_shaping[policy_id] = json_params
            else:
                self.policy_reward_shaping[policy_id] = copy.deepcopy(self.default_reward_shaping)
                if policy_id > 0:  # keep one policy with default settings in the beginning
                    log.debug('Initial rewards mutation for policy %d', policy_id)
                    self.policy_reward_shaping[policy_id] = self._perturb_reward(self.policy_reward_shaping[policy_id])

        # send initial configuration to the system components
        for policy_id in range(self.cfg.num_policies):
            self._save_cfg(policy_id)
            self._save_reward_shaping(policy_id)
            self._learner_update_cfg(policy_id)
            self._actors_update_shaping_scheme(policy_id)

    def _save_cfg(self, policy_id):
        policy_cfg_filename = policy_cfg_file(self.cfg, policy_id)
        with open(policy_cfg_filename, 'w') as json_file:
            log.debug('Saving policy-specific configuration %d to file %s', policy_id, policy_cfg_filename)
            json.dump(self.policy_cfg[policy_id], json_file)

    def _save_reward_shaping(self, policy_id):
        policy_reward_shaping_filename = policy_reward_shaping_file(self.cfg, policy_id)
        with open(policy_reward_shaping_filename, 'w') as json_file:
            log.debug('Saving policy-specific reward shaping %d to file %s', policy_id, policy_reward_shaping_filename)
            json.dump(self.policy_reward_shaping[policy_id], json_file)

    def _perturb_param(self, param, param_name, default_param):
        # toss a coin whether we perturb the parameter at all
        if random.random() > self.cfg.pbt_mutation_rate:
            return param

        if param != default_param and random.random() < 0.05:
            # small chance to replace parameter with a default value
            log.debug('%s changed to default value %r', param_name, default_param)
            return default_param

        if param_name in SPECIAL_PERTURBATION:
            new_value = SPECIAL_PERTURBATION[param_name](param, self.cfg)
        elif type(param) is bool:
            new_value = not param
        elif isinstance(param, numbers.Number):
            perturb_amount = random.uniform(1.01, 1.5)
            new_value = perturb_float(float(param), perturb_amount=perturb_amount)
        else:
            raise RuntimeError('Unsupported parameter type')

        log.debug('Param %s changed from %.6f to %.6f', param_name, param, new_value)
        return new_value

    def _perturb(self, old_params, default_params, has_inference=False):
        """Params assumed to be a flat dict."""
        params = copy.deepcopy(old_params)

        for key, value in params.items():
            if isinstance(value, (tuple, list)):
                # this is the case for reward shaping delta params
                params[key] = tuple(
                    self._perturb_param(p, f'{key}_{i}', default_params[key][i])
                    for i, p in enumerate(value)
                )
            else:
                params[key] = self._perturb_param(value, key, default_params[key])
        if not has_inference and 'full_config' in params:
            change_to = int(self._perturb_param(params['full_config']['environment']['grid_config']['size'], 'gridsize', default_params['full_config']['environment']['grid_config']['size']))
            log.info(f"Changing env size from {params['full_config']['environment']['grid_config']['size']} to {change_to}")
            try:
                with open('/home/pe/Downloads/pogema-appo-main/changelog.log') as fin:
                    lines = fin.readlines()
            except:
                lines = []
            with open('/home/pe/Downloads/pogema-appo-main/changelog.log', 'w') as fout:
                fout.write('\n'.join(lines + [f"Changing env size from {params['full_config']['environment']['grid_config']['size']} to {change_to}"]))
            params['full_config']['environment']['grid_config']['size'] = change_to
            params['full_config']['environment']['grid_config']['density'] = np.random.randint(1, 8) / 10
            if random.random() > self.cfg.pbt_mutation_rate:
                params['full_config']['environment']['grid_config']['num_agents'] = perturb_float(
                    params['full_config']['environment']['grid_config']['num_agents'], 1.5)
            if params['full_config']['environment']['grid_config']['size'] < 4 :
                params['full_config']['environment']['grid_config']['size'] = 4
            if params['full_config']['environment']['grid_config']['num_agents'] > (params['full_config']['environment']['grid_config']['size']**2 * (1-params['full_config']['environment']['grid_config']['density'])) // 2:
                params['full_config']['environment']['grid_config']['num_agents'] = (params['full_config']['environment']['grid_config']['size']**2 * (1-params['full_config']['environment']['grid_config']['density'])) // 2
        return params

    def _perturb_cfg(self, original_cfg, has_inference=False):
        replacement_cfg = copy.deepcopy(original_cfg)
        return self._perturb(replacement_cfg, default_params=self.cfg, has_inference=has_inference)

    def _perturb_reward(self, original_reward_shaping):
        if original_reward_shaping is None:
            return None

        replacement_shaping = copy.deepcopy(original_reward_shaping)

        if len(self.reward_categories_to_tune) > 0:
            for category in self.reward_categories_to_tune:
                if category in replacement_shaping:
                    replacement_shaping[category] = self._perturb(
                        replacement_shaping[category], default_params=self.default_reward_shaping[category],
                    )
        else:
            replacement_shaping = self._perturb(replacement_shaping, default_params=self.default_reward_shaping)

        return replacement_shaping

    def _force_learner_to_save_model(self, policy_id, best=False):
        learner_worker = self.learner_workers[policy_id]
        if best:
            pid = learner_worker.policy_id
            learner_worker.policy_id = -1
        learner_worker.save_model()
        if best:
            learner_worker.policy_id = pid

    def save_best(self, policy_id):
        learner_worker = self.learner_workers[policy_id]
        learner_worker.save_best_model()

    def _learner_load_model(self, policy_id, replacement_policy):
        log.debug('Asking learner %d to load model from %d', policy_id, replacement_policy)

        load_task = (PbtTask.LOAD_MODEL, (policy_id, replacement_policy))
        learner_worker = self.learner_workers[policy_id]
        learner_worker.task_queue.put((TaskType.PBT, load_task))

    def _learner_update_cfg(self, policy_id):
        learner_worker = self.learner_workers[policy_id]

        log.debug('Sending learning configuration to learner %d...', policy_id)
        cfg_task = (PbtTask.UPDATE_CFG, (policy_id, self.policy_cfg[policy_id]))
        learner_worker.task_queue.put((TaskType.PBT, cfg_task))

    def _actors_update_shaping_scheme(self, policy_id):
        log.debug('Sending latest reward scheme to actors for policy %d...', policy_id)
        for actor_worker in self.actor_workers:
            reward_scheme_task = (PbtTask.UPDATE_REWARD_SCHEME, (policy_id, self.policy_reward_shaping[policy_id]))
            task = (TaskType.PBT, reward_scheme_task)
            try:
                actor_worker.task_queue.put(task, timeout=0.1)
            except Full:
                log.warning('Could not add task %r to queue, it is likely that worker died', task)

    @staticmethod
    def _write_dict_summaries(dictionary, writer, name, env_steps):
        for d, key, value in iterate_recursively(dictionary):
            if isinstance(value, bool):
                value = int(value)

            if isinstance(value, (int, float)):
                writer.add_scalar(f'zz_pbt/{name}_{key}', value, env_steps)
            elif isinstance(value, (tuple, list)):
                for i, tuple_value in enumerate(value):
                    writer.add_scalar(f'zz_pbt/{name}_{key}_{i}', tuple_value, env_steps)
            else:
                log.error('Unsupported type in pbt summaries %r', type(value))

    def _write_pbt_summaries(self, policy_id, env_steps):
        writer = self.summary_writers[policy_id]
        self._write_dict_summaries(self.policy_cfg[policy_id], writer, 'cfg', env_steps)
        if self.policy_reward_shaping[policy_id] is not None:
            self._write_dict_summaries(self.policy_reward_shaping[policy_id], writer, 'rew', env_steps)

    def _update_policy(self, policy_id, policy_stats):
        log.info('###########################update policy#################################')
        if self.cfg.pbt_target_objective not in policy_stats:
            return
        log.info('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        target_objectives = policy_stats[self.cfg.pbt_target_objective]

        # not enough data to perform PBT yet
        for objectives in target_objectives:
            if len(objectives) <= 0:
                return

        target_objectives = [np.mean(o) for o in target_objectives]

        policies = list(range(self.cfg.num_policies))
        policies_sorted = sorted(zip(target_objectives, policies), reverse=True)
        policies_sorted = [p for objective, p in policies_sorted]

        replace_fraction = self.cfg.pbt_replace_fraction
        replace_number = math.ceil(replace_fraction * self.cfg.num_policies)

        best_policies = policies_sorted[:replace_number]
        worst_policies = policies_sorted[-replace_number:]

        if policy_id in best_policies:
            if policy_id == policies_sorted[-1]:
                self._force_learner_to_save_model(policy_id, True)
            # don't touch the policies that are doing well
            return

        log.debug('PBT best policies: %r, worst policies %r', best_policies, worst_policies)

        # to make the code below uniform, this means keep our own parameters and cfg
        # we only take parameters and cfg from another policy if certain conditions are met (see below)
        replacement_policy = policy_id

        if policy_id in worst_policies:
            log.debug('Current policy %d is among the worst policies %r', policy_id, worst_policies)

            replacement_policy_candidate = random.choice(best_policies)
            reward_delta = target_objectives[replacement_policy_candidate] - target_objectives[policy_id]
            reward_delta_relative = abs(reward_delta / (target_objectives[replacement_policy_candidate] + EPS))  # TODO: this might not work correctly with negative rewards

            if abs(reward_delta) > self.cfg.pbt_replace_reward_gap_absolute and reward_delta_relative > self.cfg.pbt_replace_reward_gap:
                replacement_policy = replacement_policy_candidate
                log.debug(
                    'Difference in reward is %.4f (%.4f), policy %d weights to be replaced by %d',
                    reward_delta, reward_delta_relative, policy_id, replacement_policy,
                )
            else:
                log.debug('Difference in reward is not enough %.3f %.3f', abs(reward_delta), reward_delta_relative)
        has_inference = np.any([runner.worker_idx + runner.split_idx == 0 for runner in self.learner_workers[policy_id].env_runners])
        if policy_id == 0:
            # Do not ever mutate the 1st policy, leave it for the reference
            # Still we allow replacements in case it's really bad
            self.policy_cfg[policy_id] = self.policy_cfg[replacement_policy]
            self.policy_reward_shaping[policy_id] = self.policy_reward_shaping[replacement_policy]
        else:
            self.policy_cfg[policy_id] = self._perturb_cfg(self.policy_cfg[replacement_policy], has_inference)
            self.policy_reward_shaping[policy_id] = self._perturb_reward(self.policy_reward_shaping[replacement_policy])

        skipfirst = 0 if not has_inference else 1
        for i,runner in enumerate(self.learner_workers[policy_id].env_runners):
            if runner.worker_idx + runner.split_idx != 0:
                self.learner_workers[policy_id].env_runners[i].close()

        for split_idx in range(skipfirst,self.learner_workers[policy_id].num_splits):
            env_runner = VectorEnvRunner(
                self.learner_workers[policy_id].cfg, self.learner_workers[policy_id].vector_size // self.learner_workers[policy_id].num_splits, self.learner_workers[policy_id].worker_idx, split_idx, self.learner_workers[policy_id].num_agents,
                self.learner_workers[policy_id].shared_buffers, self.learner_workers[policy_id].reward_shaping,
            )
            env_runner.init(start=False)
            self.learner_workers[policy_id].env_runners[split_idx] = env_runner

        if replacement_policy != policy_id:
            # force replacement policy learner to save the model and wait until it's done
            self._force_learner_to_save_model(replacement_policy)

            # now that the latest "replacement" model is saved to disk, we ask the learner to load the replacement policy
            self._learner_load_model(policy_id, replacement_policy)

        self._save_cfg(policy_id)
        self._save_reward_shaping(policy_id)
        self._learner_update_cfg(policy_id)
        self._actors_update_shaping_scheme(policy_id)

    def update(self, env_steps, policy_stats):
        if not self.cfg.with_pbt or self.cfg.num_policies <= 1:
            return
        log.info('**************************before loop policy************************')
        for policy_id in range(self.cfg.num_policies):
            if policy_id not in env_steps:
                continue

            if env_steps[policy_id] < self.cfg.pbt_start_mutation:
                continue

            steps_since_last_update = env_steps[policy_id] - self.last_update[policy_id]
            log.info('$$$$$$$$$$$$$$$$$$$$$$$before if policy$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
            if steps_since_last_update > self.cfg.pbt_period_env_steps:
                log.info('!!!!!!!!!!!!!!!!!!!!!!!before update policy!!!!!!!!!!!!!!!!!!!!!!!!')
                self._update_policy(policy_id, policy_stats)
                self._write_pbt_summaries(policy_id, env_steps[policy_id])
                self.last_update[policy_id] = env_steps[policy_id]

        # also periodically dump a pbt summary even if we didn't change anything
        now = time.time()
        if now - self.last_pbt_summaries > 5 * 60:
            for policy_id in range(self.cfg.num_policies):
                if policy_id in env_steps:
                    self._write_pbt_summaries(policy_id, env_steps[policy_id])
                    self.last_pbt_summaries = now

#-------------------------------------------------------------------------------------------------------------------------



class VectorEnvRunner:
    """
    A collection of environments simulated sequentially.
    With double buffering each actor worker holds two vector runners and switches between them.
    Without single buffering we only use a single VectorEnvRunner per actor worker.

    All envs on a single VectorEnvRunner run in unison, e.g. they all do one step at a time together.
    This also means they all finish their rollouts together. This allows us to minimize the amount of messages
    passed around.

    Individual envs (or agents in these envs in case of multi-agent) can potentially be controlled by different
    policies when we're doing PBT. We only start simulating the next step in the environment when
    all actions from all envs and all policies are collected. This leaves optimization potential: we can start
    simulating some envs right away as actions for them arrive. But usually double-buffered sampling masks
    this type of inefficiency anyway. The worker is probably still rendering a previous vector of envs when
    the actions arrive.
    """

    def __init__(self, cfg, num_envs, worker_idx, split_idx, num_agents, shared_buffers, pbt_reward_shaping):
        """
        Ctor.

        :param cfg: global system config (all CLI params)
        :param num_envs: number of envs to run in this vector runner
        :param worker_idx: idx of the parent worker
        :param split_idx: index of the environment group in double-buffered sampling (either 0 or 1). Always 0 when
        double-buffered sampling is disabled.
        :param num_agents: number of agents in each env (1 for single-agent envs)
        :param shared_buffers: a collection of all shared data structures used by the algorithm. Most importantly,
        the trajectory buffers in shared memory.
        :param pbt_reward_shaping: initial reward shaping dictionary, for configuration where PBT optimizes
        reward coefficients in environments.
        """

        self.cfg = cfg

        self.num_envs = num_envs
        self.worker_idx = worker_idx
        self.split_idx = split_idx

        self.rollout_step = 0

        self.num_agents = num_agents  # queried from env

        self.shared_buffers = shared_buffers
        self.policy_output_tensors = self.shared_buffers.policy_output_tensors[self.worker_idx, self.split_idx]

        self.envs, self.actor_states, self.episode_rewards = [], [], []

        self.pbt_reward_shaping = pbt_reward_shaping

        self.policy_mgr = PolicyManager(self.cfg, self.num_agents)

    def init(self, start = True):
        """
        Actually instantiate the env instances.
        Also creates ActorState objects that hold the state of individual actors in (potentially) multi-agent envs.
        """

        for env_i in range(self.num_envs):
            vector_idx = self.split_idx * self.num_envs + env_i

            # global env id within the entire system
            env_id = self.worker_idx * self.cfg.num_envs_per_worker + vector_idx

            env_config = AttrDict(
                worker_index=self.worker_idx, vector_index=vector_idx, env_id=env_id,
            )

            log.info('Creating env %r... %d-%d-%d', env_config, self.worker_idx, self.split_idx, env_i)
            env = make_env_func(self.cfg, env_id, env_config=env_config, start=start)
            log.info(str(self.cfg))
            print(self.cfg)
            with open('/home/pe/Downloads/pogema-appo-main/cfg.txt', 'w') as fout:
                fout.write(str(self.cfg))
            env.seed(env_id)
            self.envs.append(env)

            actor_states_env, episode_rewards_env = [], []
            for agent_idx in range(self.num_agents):
                actor_state = ActorState(
                    self.cfg, env, self.worker_idx, self.split_idx, env_i, agent_idx,
                    self.shared_buffers, self.policy_output_tensors[env_i, agent_idx],
                    self.pbt_reward_shaping, self.policy_mgr,
                )
                actor_states_env.append(actor_state)
                episode_rewards_env.append(0.0)

            self.actor_states.append(actor_states_env)
            self.episode_rewards.append(episode_rewards_env)

    def update_env_steps(self, env_steps):
        for env_i in range(self.num_envs):
            for agent_i in range(self.num_agents):
                self.actor_states[env_i][agent_i].approx_env_steps = env_steps

    def _process_policy_outputs(self, policy_id, timing):
        """
        Process the latest data from the policy worker (for policy = policy_id).
        Policy outputs currently include new RNN states, actions, values, logprobs, etc. See shared_buffers.py
        for the full list of outputs.

        As a performance optimization, all these tensors are squished together into a single tensor.
        This allows us to copy them to shared memory only once, which makes a difference on the policy worker.
        Here we do np.split to separate them back into individual tensors.

        :param policy_id: index of the policy whose outputs we're currently processing
        :return: whether we got all outputs for all the actors in our VectorEnvRunner. If this is True then we're
        ready for the next step of the simulation.
        """

        all_actors_ready = True

        for env_i in range(self.num_envs):
            for agent_i in range(self.num_agents):
                actor_state = self.actor_states[env_i][agent_i]
                if not actor_state.is_active:
                    continue

                actor_policy = actor_state.curr_policy_id
                assert actor_policy != -1

                if actor_policy == policy_id:
                    # via shared memory mechanism the new data should already be copied into the shared tensors

                    with timing.add_time('split_output_tensors'):
                        policy_outputs = np.split(
                            actor_state.policy_output_tensors,
                            indices_or_sections=actor_state.policy_output_indices,
                            axis=0,
                        )
                    policy_outputs_dict = dict()
                    new_rnn_state = None
                    for tensor_idx, name in enumerate(actor_state.policy_output_names):
                        if name == 'rnn_states':
                            new_rnn_state = policy_outputs[tensor_idx]
                        else:
                            policy_outputs_dict[name] = policy_outputs[tensor_idx]

                    # save parsed trajectory outputs directly into the trajectory buffer
                    actor_state.set_trajectory_data(policy_outputs_dict, self.rollout_step)
                    actor_state.last_actions = policy_outputs_dict['actions']

                    # this is an rnn state for the next iteration in the rollout
                    actor_state.last_rnn_state = new_rnn_state

                    actor_state.ready = True
                elif not actor_state.ready:
                    all_actors_ready = False

        # Potential optimization: when actions are ready for all actors within one environment we can execute
        # a simulation step right away, without waiting for all other actions to be calculated.
        return all_actors_ready

    def _process_rewards(self, rewards, env_i):
        """
        Pretty self-explanatory, here we record the episode reward and apply the optional clipping and
        scaling of rewards.
        """

        for agent_i, r in enumerate(rewards):
            self.actor_states[env_i][agent_i].last_episode_reward += r

        rewards = np.asarray(rewards, dtype=np.float32)
        rewards = rewards * self.cfg.reward_scale
        rewards = np.clip(rewards, -self.cfg.reward_clip, self.cfg.reward_clip)
        return rewards

    def _process_env_step(self, new_obs, rewards, dones, infos, env_i):
        """
        Process step outputs from a single environment in the vector.

        :param new_obs: latest observations from the env
        :param env_i: index of the environment in the vector
        :return: episodic stats, not empty only on the episode boundary
        """

        episodic_stats = []
        env_actor_states = self.actor_states[env_i]

        rewards = self._process_rewards(rewards, env_i)

        for agent_i in range(self.num_agents):
            actor_state = env_actor_states[agent_i]

            episode_report = actor_state.record_env_step(
                rewards[agent_i], dones[agent_i], infos[agent_i], self.rollout_step,
            )

            actor_state.last_obs = new_obs[agent_i]
            actor_state.update_rnn_state(dones[agent_i])

            if episode_report:
                episodic_stats.append(episode_report)

        return episodic_stats

    def _finalize_trajectories(self):
        """
        Do some postprocessing when we're done with the rollout.
        """

        rollouts = []
        for env_i in range(self.num_envs):
            for agent_i in range(self.num_agents):
                rollouts.extend(self.actor_states[env_i][agent_i].finalize_trajectory(self.rollout_step))

        return rollouts

    def _update_trajectory_buffers(self, timing):
        """
        Request free trajectory buffers to store the next rollout.
        """
        num_buffers_to_request = 0
        for env_i in range(self.num_envs):
            for agent_i in range(self.num_agents):
                num_buffers_to_request += self.actor_states[env_i][agent_i].needs_buffer

        if num_buffers_to_request > 0:
            traj_buffer_indices = self.shared_buffers.get_trajectory_buffers(num_buffers_to_request, timing)

            i = 0
            for env_i in range(self.num_envs):
                for agent_i in range(self.num_agents):
                    actor_state = self.actor_states[env_i][agent_i]
                    if actor_state.needs_buffer:
                        buffer_idx = traj_buffer_indices[i]
                        actor_state.update_traj_buffer(buffer_idx)
                        i += 1

    def _format_policy_request(self):
        """
        Format data that allows us to request new actions from policies that control the agents in all the envs.
        Note how the data required is basically just indices of envs and agents, as well as location of the step
        data in the shared rollout buffer. This is enough for the policy worker to find the step data in the shared
        data structure.

        :return: formatted request to be distributed to policy workers through FIFO queues.
        """

        policy_request = dict()

        for env_i in range(self.num_envs):
            for agent_i in range(self.num_agents):
                actor_state = self.actor_states[env_i][agent_i]

                if actor_state.is_active:
                    policy_id = actor_state.curr_policy_id

                    # where policy worker should look for the policy inputs for the next step
                    data = (env_i, agent_i, actor_state.curr_traj_buffer_idx, self.rollout_step)

                    if policy_id not in policy_request:
                        policy_request[policy_id] = []
                    policy_request[policy_id].append(data)

        return policy_request

    def _prepare_next_step(self):
        """
        Write environment outputs to shared memory so policy workers can calculate actions for the next step.
        Note how we temporarily hold obs and rnn_states in local variables before writing them into shared memory.
        We could not do the memory write right away because for that we need the memory location of the NEXT step.
        If this is the first step in the new rollout, we need to switch to a new trajectory buffer before we do that
        (because the previous trajectory buffer is now used by the learner and we can't use it until the learner is
        done).
        """

        for env_i in range(self.num_envs):
            for agent_i in range(self.num_agents):
                actor_state = self.actor_states[env_i][agent_i]

                if actor_state.is_active:
                    actor_state.ready = False

                    # populate policy inputs in shared memory
                    policy_inputs = dict(obs=actor_state.last_obs, rnn_states=actor_state.last_rnn_state)
                    actor_state.set_trajectory_data(policy_inputs, self.rollout_step)
                else:
                    actor_state.ready = True

    def reset(self, report_queue):
        """
        Do the very first reset for all environments in a vector. Populate shared memory with initial obs.
        Note that this is called only once, at the very beginning of training. After this the envs should auto-reset.

        :param report_queue: we use report queue to monitor reset progress (see appo.py). This can be a lengthy
        process.
        :return: first requests for policy workers (to generate actions for the very first env step)
        """

        for env_i, e in enumerate(self.envs):
            observations = e.reset()

            if self.cfg.decorrelate_envs_on_one_worker:
                env_i_split = self.num_envs * self.split_idx + env_i
                decorrelate_steps = self.cfg.rollout * env_i_split + self.cfg.rollout * random.randint(0, 4)

                log.info('Decorrelating experience for %d frames...', decorrelate_steps)
                for decorrelate_step in range(decorrelate_steps):
                    actions = [e.action_space.sample() for _ in range(self.num_agents)]
                    observations, rew, dones, info = e.step(actions)

            for agent_i, obs in enumerate(observations):
                actor_state = self.actor_states[env_i][agent_i]
                actor_state.set_trajectory_data(dict(obs=obs), self.rollout_step)
                # rnn state is already initialized at zero

            safe_put(report_queue, dict(initialized_env=(self.worker_idx, self.split_idx, env_i)), queue_name='report')

        policy_request = self._format_policy_request()
        return policy_request

    def advance_rollouts(self, data, timing):
        """
        Main function in VectorEnvRunner. Does one step of simulation (if all actions for all actors are available).

        :param data: incoming data from policy workers (policy outputs), including new actions
        :param timing: this is just for profiling
        :return: same as reset(), return a set of requests for policy workers, asking them to generate actions for
        the next env step.
        """

        with timing.add_time('save_policy_outputs'):
            policy_id = data['policy_id']
            all_actors_ready = self._process_policy_outputs(policy_id, timing)
            if not all_actors_ready:
                # not all policies involved sent their actions, waiting for more
                return None, None, None

        complete_rollouts, episodic_stats = [], []

        for env_i, e in enumerate(self.envs):
            with timing.add_time('env_step'):
                actions = [s.curr_actions() for s in self.actor_states[env_i]]
                new_obs, rewards, dones, infos = e.step(actions)

            with timing.add_time('overhead'):
                stats = self._process_env_step(new_obs, rewards, dones, infos, env_i)
                episodic_stats.extend(stats)

        self.rollout_step += 1
        if self.rollout_step == self.cfg.rollout:
            # finalize and serialize the trajectory if we have a complete rollout
            complete_rollouts = self._finalize_trajectories()
            self._update_trajectory_buffers(timing)
            self.rollout_step = 0

        with timing.add_time('prepare_next_step'):
            self._prepare_next_step()

        policy_request = self._format_policy_request()

        return policy_request, complete_rollouts, episodic_stats

    def close(self):
        for e in self.envs:
            e.close()