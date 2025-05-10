import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.callbacks import BaseCallback
from teacher_config import REWARD_WEIGHTS,OTHER_PARAMS

# ─── OBSERVATION WRAPPER ─────────────────────────────────────────────────────
class PrivilegedObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        orig = env.observation_space
        # orig = env._get_obs()
        lows, highs = [], []
        for key in [
            "joint_positions", "joint_velocities",
            "block_positions", "target_positions",
            "gripper_positions"
        ]:
            lows.append(orig[key].low)
            highs.append(orig[key].high)
        self.observation_space = spaces.Box(
            np.concatenate(lows),
            np.concatenate(highs),
            dtype=np.float32
        )
    def observation(self, obs):
        return np.concatenate([
            obs["joint_positions"],
            obs["joint_velocities"],
            obs["block_positions"],
            obs["target_positions"],
            obs["gripper_positions"],
        ])

# ─── DYNAMIC REWARD FUNCTIONS ─────────────────────────────────────────────────
# TODO: make info[] work
def reward_reach(info,obs):
    # negative distance of closest arm to object
    # return -info['closest_dist']
    left_gripper_pos = obs["gripper_positions"][0:3]
    right_gripper_pos = obs["gripper_positions"][3:6]
    block_pos = obs["block_positions"]
    # print(left_gripper_pos.shape)
    # import ipdb;ipdb.set_trace() # TODO check how many env there are, if so may need dim=1

    distance = min(np.linalg.norm(left_gripper_pos - block_pos),np.linalg.norm(right_gripper_pos - block_pos))
    return 1 - np.tanh(distance / OTHER_PARAMS["reach_std"])
def reward_lift(info, obs):
    # keep the block lifted if the distance between block and target not small enough
    distance_target = np.linalg.norm(obs["block_positions"][0:2]-obs["target_positions"][0:2])
    close_enough = np.where(distance_target > OTHER_PARAMS["putdown_thresh"], 1.0, 0.0)
    return np.where(obs["block_positions"][2] > OTHER_PARAMS["minimal_height"], 1.0, 0.0)*close_enough

def reward_grasp_slip(info,obs):
    # negative slip distance between gripper and object
    return -np.linalg.norm(info['object_pos'] - info['gripper_pos'])

def reward_grasp_close(info,obs):
    # encourage closing when at object
    return float(info.get('at_object', 0.0))


def reward_transport(info,obs):
    # only care about 2d accuracy
    distance_target = np.linalg.norm(obs["block_positions"][0:2]-obs["target_positions"][0:2])
    close_enough = np.where(distance_target > OTHER_PARAMS["putdown_thresh"], 1.0, 0.0)
    not_close_distance_rew =  (1-close_enough)*obs["block_positions"][2] > OTHER_PARAMS["minimal_height"]* (1 - np.tanh(distance_target / OTHER_PARAMS["transport_std"]))
    return not_close_distance_rew
def reward_transport_putdown(info,obs):
    # add 3d accuracy to put down
    distance_target2D = np.linalg.norm(obs["block_positions"][0:2]-obs["target_positions"][0:2])
    distance_target = np.linalg.norm(obs["block_positions"]-obs["target_positions"])
    close_enough = np.where(distance_target2D > OTHER_PARAMS["putdown_thresh"], 1.0, 0.0)
    putdown_rew =  close_enough*(1 - np.tanh(distance_target / OTHER_PARAMS["transport_putdown_std"]))
    return putdown_rew

def reward_action_penalty(info,obs):
    return np.sum(np.square(obs["cur_action"]-obs["prev_action"]))

def reward_joint_vel_panelty(info,obs):
    # return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
    return np.sum(np.square(obs["joint_velocities"]))
def reward_place_success(info,obs):
    # sparse success indicator
    return float(info.get('placed_successfully', 0.0))

def reward_place_accuracy(info,obs): # this reward is problematic
    # negative final placement error
    if info.get('placed_successfully', False):
        return -np.linalg.norm(info['object_pos'] - info['target_pos'])
    return 0.0

def reward_action_penalty(info,obs):
    # penalize large actions and joint velocities
    a = info.get('action', np.zeros(1))
    v = info.get('joint_vel', np.zeros(1))
    return -(np.sum(a**2) + np.sum(v**2))

def reward_closest_arm(info,obs):
    # reward if correct (closest) arm was used
    return float(info.get('used_closest', 0.0))

def reward_far_arm_penalty(info,obs):
    # penalty if the other arm moved
    return -float(info.get('other_moved', 0.0))

class RewardShapingWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, _: float) -> float:
        info = self.env._get_info()
        obs = self.env._get_obs()
        # dynamic composition of rewards
        total_r = 0.0
        for name, weight in REWARD_WEIGHTS.items():
            if weight <= 0.0:
                continue
            func = globals().get("reward_"+name)
            if func is None:
                raise ValueError(f"Reward function '{name}' is not defined")
            total_r += weight * func(info, obs)
        return total_r

# ─── CALLBACKS FACTORY ─────────────────────────────────────────────────────────
class EpisodeRewardCallback(BaseCallback):
    """
    At the end of each rollout (i.e. each PPO iteration), print
    the average return over all episodes in ep_info_buffer.
    """
    def _on_step(self) -> bool:
        # This is called at every env.step() and MUST be implemented.
        return True

    def _on_rollout_end(self) -> None:
        # This is called once per PPO update (end of rollout).
        rewards = [info["r"] for info in self.model.ep_info_buffer]
        if rewards:
            avg = np.mean(rewards)
            print(f"→ Avg episode reward (last {len(rewards)} eps): {avg:.2f}")

def make_callbacks(eval_env, train_config, savepath):
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=savepath,
        log_path=savepath,
        eval_freq=train_config["eval_freq"],
        n_eval_episodes=train_config["eval_episodes"],
        deterministic=True,
        render=False,
    )
    chkpt_cb = CheckpointCallback(
        save_freq=train_config["save_freq"],
        save_path=savepath,
        name_prefix="ppo_teacher"
    )
    return [eval_cb, chkpt_cb]
