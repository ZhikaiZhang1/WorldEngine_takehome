import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.callbacks import BaseCallback
from teacher_config import REWARD_WEIGHTS,OTHER_PARAMS, PPO_PARAMS

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

# ─── REWARD FUNCTIONS ─────────────────────────────────────────────────
def choose_gripper(info, obs):
    left_base, right_base = info["base_positions"][:3], info["base_positions"][3:6]
    left_grip,  right_grip  = obs["gripper_positions"][:3], obs["gripper_positions"][3:6]
    block_pos = obs["block_positions"]
    block_pos_init = info["block_pos_init"]

    # choose finger
    d_left  = np.linalg.norm(block_pos_init - left_base)
    d_right = np.linalg.norm(block_pos_init - right_base)
    
    if d_left < d_right:
        grip = left_grip; reachDist = np.linalg.norm(block_pos - left_grip)
        init_grip = info["gripper_pos_init"][0:3]
        actiongrip = obs["joint_positions"][6:8]
        arm = "left"

    else:
        grip = right_grip; reachDist = np.linalg.norm(block_pos - right_grip)
        init_grip = info["gripper_pos_init"][3:6]
        actiongrip = obs["joint_positions"][14:16]
        arm = "right"

    # print("distance", reachDist,"left", np.linalg.norm(block_pos - left_grip), "right", np.linalg.norm(block_pos - right_grip))
    
    return grip, init_grip, actiongrip, reachDist,arm
def left_gripper(info, obs):
    left_base, right_base = info["base_positions"][:3], info["base_positions"][3:6]
    left_grip,  right_grip  = obs["gripper_positions"][:3], obs["gripper_positions"][3:6]
    block_pos = obs["block_positions"]
    block_pos_init = info["block_pos_init"]

    grip = left_grip; reachDist = np.linalg.norm(block_pos - left_grip)
    init_grip = info["gripper_pos_init"][0:3]
    actiongrip = obs["joint_positions"][6:8]
    arm = "left"
    return grip, init_grip, actiongrip, reachDist,arm

def was_lifted(info):
    return info["prev_block_pos"][2] > OTHER_PARAMS["minimal_height"]

def lifted(reachDist, block_pos):
    # if reachDist is small, and object is higher than thresh, then it has been lifted
    if reachDist < OTHER_PARAMS["close_thresh"] and block_pos[2] > OTHER_PARAMS["minimal_height"]:
        return True
    return False
def dropped(reachDist, block_pos, gripper_pos):
    return reachDist > OTHER_PARAMS["close_thresh"] and block_pos[2] < OTHER_PARAMS["minimal_height"] and gripper_pos[2] > OTHER_PARAMS["minimal_height"]

def reward_pick_stage(info, obs):
    # return a large reward if object lifted bu previously not
    block_pos_init = info["block_pos_init"]
    block_pos = obs["block_positions"]
    grip, init_grip, actiongrip,reachDist, arm_choice = choose_gripper(info, obs)
    if lifted(reachDist, block_pos) and not was_lifted(info):
        return 1.0*OTHER_PARAMS["minimal_height"]
    elif block_pos[2]>block_pos_init[2] and reachDist < OTHER_PARAMS["close_thresh"]:
        return block_pos[2]
    return 0.0

def reward_drop_panelty_stage(info, obs):
    grip, init_grip, actiongrip,reachDist, arm_choice = choose_gripper(info, obs)
    if was_lifted(info) and dropped(reachDist,obs["block_positions"],grip):
        return -1
    return 0
def reward_place_stage(info, obs):
    c1 = 1000
    c2 = 0.01
    c3 = 0.001

    grip, init_grip, actiongrip,reachDist, arm_choice = choose_gripper(info, obs)
    cond = was_lifted(info) and OTHER_PARAMS["close_thresh"] and not (dropped(reachDist,obs["block_positions"],grip))
    if cond:
        placeRew = 1000 * (OTHER_PARAMS["maxPlacingDist"] - placingDist) + c1 * (
            np.exp(-(placingDist**2) / c2)
            + np.exp(-(placingDist**2) / c3)
        )
        placeRew = max(placeRew, 0)
        return placeRew
    else:
        return 0
    return


def reward_reach(info,obs):
    # linear reward for farther, and tanh for closer
    # grip, init_grip, actiongrip,reachDist, arm_choice = choose_gripper(info, obs)
    grip, init_grip, actiongrip,reachDist, arm_choice = left_gripper(info, obs)

    
    block_pos = obs["block_positions"]

    reachDistXY = np.linalg.norm(block_pos[:2] - grip[:2])
    init_grip_pos = info["gripper_pos_init"]
    init_reachDistXY = np.linalg.norm(block_pos[:2] - init_grip[:2])
    init_reachDist = np.linalg.norm(block_pos - init_grip)
    
    block_pos_init = info["block_pos_init"]

    block_not_lifted = np.where(obs["block_positions"][2] < (OTHER_PARAMS["minimal_height"]+block_pos_init[2]), 1.0, 0.0)
    # if reachDistXY < OTHER_PARAMS["reach_thresh"]:
    #     distance = reachDist
    # else:
    #     distance = reachDistXY
    distance = reachDist


    # if distance > 0.3:
    #     reward = (OTHER_PARAMS["lin2tanh_connect"] - (distance / (init_reachDist/OTHER_PARAMS["lin2tanh_connect"])))*block_not_lifted
    # else:
    #     reward = (1 - np.tanh(distance / OTHER_PARAMS["reach_std"]))*block_not_lifted
    # reward = (1 - np.tanh(distance / OTHER_PARAMS["reach_std"]))*block_not_lifted
    reward = (np.power(2,-distance/(OTHER_PARAMS["reach_std"]*3)))*block_not_lifted
    

    if distance < OTHER_PARAMS["close_thresh"]:
        reward += max(actiongrip, 0) / OTHER_PARAMS["grasp_scale"]
    
    return reward

def reward_distance(info,obs):
    # linear reward for farther, and tanh for closer
    # grip, init_grip, actiongrip,reachDist, arm_choice = choose_gripper(info, obs)
    grip, init_grip, actiongrip,reachDist, arm_choice = left_gripper(info, obs)

    
    return -reachDist

def reward_wrong_arm_panelty(info,obs):
    grip, init_grip, actiongrip,reachDist, arm_choice = choose_gripper(info, obs)
    if arm_choice == "left":
        right_init_grip_pos = info["gripper_pos_init"][3:6]
        right_grip  = obs["gripper_positions"][3:6]
        distance = np.linalg.norm(right_grip-right_init_grip_pos)
        return -(1-np.exp(-distance/OTHER_PARAMS["act_penalty_std"]))
    if arm_choice == "right":
        left_init_grip_pos = info["gripper_pos_init"][0:3]
        left_grip  = obs["gripper_positions"][0:3]
        distance = np.linalg.norm(left_grip-left_init_grip_pos)
        return -(1-np.exp(-distance/OTHER_PARAMS["act_penalty_std"]))



def reward_reach_inv_sq(info,obs):
    # negative distance of closest arm to object
    # return -info['closest_dist']
    left_base_pos = info["base_positions"][0:3]
    right_base_pos = info["base_positions"][3:6]
    left_gripper_pos = obs["gripper_positions"][0:3]
    right_gripper_pos = obs["gripper_positions"][3:6]
    block_pos = obs["block_positions"]
    block_pos_init = info["block_pos_init"]
    # print(left_base_pos.shape)
    # import ipdb;ipdb.set_trace() # TODO check how many env there are, if so may need dim=1
    choice = "left"
    distance = np.linalg.norm(left_gripper_pos[:2] - block_pos[:2])
    left_closer = np.linalg.norm(left_base_pos[0:2] - block_pos_init[0:2])>np.linalg.norm(right_base_pos[0:2] - block_pos_init[0:2])
    if not left_closer:
        distance = np.linalg.norm(right_gripper_pos[:2] - block_pos[:2])
        choice = "right"

    block_not_lifted = np.where(obs["block_positions"][2] < (OTHER_PARAMS["minimal_height"]+block_pos_init[2]), 1.0, 0.0)
    reward = 1.0 / (1.0 + distance**2)
    reward = np.pow(reward, 2)
    reward2 = 2*(np.power(2,-distance/(OTHER_PARAMS["reach_std"]*2)))

    # if distance <= 0.3:
    #     if choice == "right"
    #         distance3d = 
    #     reward2 = 2*(np.power(2,-distance3d/(OTHER_PARAMS["reach_std"]*2)))
    if distance < OTHER_PARAMS["close_thresh"]:
        finger_dis = np.linalg.norm(actiongrip - np.array([0.04,0.04]))
        reward += 0.5*(np.power(2,-finger_dis/(OTHER_PARAMS["reach_std"]/10)))
        
    return np.where(distance <= 0.3, reward2, reward)*block_not_lifted

def reward_transport(info,obs):
    # only care about 2d accuracy at first, then towards 3D
    # 
    grip, init_grip, actiongrip,reachDist, arm_choice = left_gripper(info, obs)

    distance_target = np.linalg.norm(obs["block_positions"][0:2]-obs["target_positions"][0:2])
    distancez = np.linalg.norm(obs["block_positions"][2]-obs["target_positions"][2])
    close_enough = np.where(distance_target > OTHER_PARAMS["putdown_thresh"], 1.0, 0.0)
    not_close_distance_rew =  (1-close_enough)*obs["block_positions"][2] > OTHER_PARAMS["minimal_height"]* (1 - np.tanh(distance_target / OTHER_PARAMS["transport_std"]))
    
    reward = 1.0 / (1.0 + (distance_target/0.7)**2)
    reward = np.pow(reward, 2)
    reward2 = 2*(np.power(2,-distance_target/(OTHER_PARAMS["reach_std"])))
    if distance_target > OTHER_PARAMS["putdown_thresh"]:
        ret_reward = reward*np.where(lifted(reachDist,obs["block_positions"]),1.0,0.0)
    elif distance_target <= OTHER_PARAMS["putdown_thresh"]:
        ret_reward = reward2+2*(np.power(2,-distancez/(OTHER_PARAMS["reach_std"]*2)))

    return ret_reward
def reward_transport_putdown(info,obs):
    # add 3d accuracy to put down
    distance_target2D = np.linalg.norm(obs["block_positions"][0:2]-obs["target_positions"][0:2])
    distance_target = np.linalg.norm(obs["block_positions"]-obs["target_positions"])
    close_enough = np.where(distance_target2D < OTHER_PARAMS["putdown_thresh"], 1.0, 0.0)
    putdown_rew =  close_enough*(1 - np.tanh(distance_target / OTHER_PARAMS["transport_putdown_std"]))
    return putdown_rew

def gaussian_tolerance(d, bounds=(0, 0.05), margin=0.35):
    lower, upper = bounds
    if lower <= d <= upper:
        return 1.0
    else:
        # Gaussian falloff
        scale = margin / 2.0
        return np.exp(-0.5 * ((d - upper) / scale) ** 2)

def reward_gaussian_reach(info,obs):
    left_base_pos = info["base_positions"][0:3]
    right_base_pos = info["base_positions"][3:6]
    left_gripper_pos = obs["gripper_positions"][0:3]
    right_gripper_pos = obs["gripper_positions"][3:6]
    block_pos = obs["block_positions"]
    block_pos_init = info["block_pos_init"]
    distance = np.linalg.norm(left_gripper_pos - block_pos)
    left_closer = np.linalg.norm(left_base_pos[0:2] - block_pos_init[0:2])>np.linalg.norm(right_base_pos[0:2] - block_pos_init[0:2])
    # if not left_closer:
    #     distance = np.linalg.norm(right_gripper_pos - block_pos)
    block_not_lifted = np.where(obs["block_positions"][2] < (OTHER_PARAMS["minimal_height"]+block_pos_init[2]), 1.0, 0.0)
    return gaussian_tolerance(distance)*block_not_lifted


def reward_time_penalty(info,obs):

    if info.get("success", False):
        return 0.0
    return -1.0

def reward_lift(info, obs):
    # keep the block lifted if the distance between block and target not small enough
    distance_target = np.linalg.norm(obs["block_positions"][0:2]-obs["target_positions"][0:2])
    close_enough = np.where(distance_target > OTHER_PARAMS["putdown_thresh"], 1.0, 0.0)
    return np.where(obs["block_positions"][2] > OTHER_PARAMS["minimal_height"], 1.0, 0.0)*close_enough



def reward_action_penalty(info,obs):
    return np.sum(np.square(obs["cur_action"]-obs["prev_action"]))

def reward_joint_vel_panelty(info,obs):
    # return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
    return -np.sum(np.square(obs["joint_velocities"]))
def reward_place_success(info,obs):
    # sparse success indicator
    return float(info.get('success', 0.0))

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
            info["reward_"+name] = weight * func(info, obs)
        # print(info)
        
        return total_r

# ─── CALLBACKS FACTORY ─────────────────────────────────────────────────────────
class EpisodeRewardCallback(BaseCallback):
    """
    At the end of each rollout (i.e. each PPO iteration), print
    the average return over all episodes in ep_info_buffer.
    """
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.iteration = 0
    def _on_step(self) -> bool:
        # This is called at every env.step() and MUST be implemented.
        return True

    def _on_rollout_end(self) -> None:
        # This is called once per PPO update (end of rollout).
        self.iteration += 1
        rewards = [info["r"] for info in self.model.ep_info_buffer]
        if rewards:
            avg = np.mean(rewards)
            print(f"→ Avg episode reward (last {len(rewards)} eps): {avg:.2f}")
            print(f"[Iteration {self.iteration:3d}]  "
                  f"Avg episode reward (last {len(rewards)} eps): {avg:.2f}")
            self.logger.record(f"reward/mean_reward", float(avg))

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
    save_iter_cb = SaveEveryIterationCallback(
        save_every = train_config["save_freq_iters"],  # 50
        save_path  = savepath,
        verbose    = 1
    )
    stop_cb = StopAfterNIterations(max_iterations=train_config["total_iter"], verbose=1)

    tensorReward_cb = RewardComponentLogger(list(REWARD_WEIGHTS.keys()), verbose=1)
    adaptive_lr_cb = AdaptiveLrCallback(desired_kl = PPO_PARAMS["target_kl"], min_lr     = 1e-5, max_lr     = 1e-2, verbose    = 1,)
    return [eval_cb,save_iter_cb,stop_cb,tensorReward_cb,adaptive_lr_cb]


class SaveEveryIterationCallback(BaseCallback):
    """
    Save the model every `save_every` PPO iterations (rollouts).
    """
    def __init__(self, save_every: int, save_path: str, verbose=1):
        super().__init__(verbose)
        self.save_every = save_every
        self.save_path  = save_path
        self.iter_count = 0

    def _on_rollout_end(self) -> None:
        # This hook is called exactly once per PPO iteration
        self.iter_count += 1
        if self.iter_count % self.save_every == 0:
            path = f"{self.save_path}/model_iter_{self.iter_count}"
            self.model.save(path)
            if self.verbose:
                print(f"→ [Iteration {self.iter_count}] Saved model to: {path}")

    def _on_step(self) -> bool:
        # required abstract method, not used here
        return True

class StopAfterNIterations(BaseCallback):
    """
    Stop training after `max_iterations` PPO rollouts.
    """
    def __init__(self, max_iterations: int, verbose: int = 0):
        super().__init__(verbose)
        self.max_iterations = max_iterations
        self.iteration_cnt  = 0
        self._stop_now      = False

    def _on_rollout_end(self) -> None:
        # Called once per PPO iteration (end of rollout)
        self.iteration_cnt += 1
        if self.iteration_cnt >= self.max_iterations:
            if self.verbose:
                print(f"[Callback] Reached {self.max_iterations} iterations → stopping")
            self._stop_now = True

    def _on_step(self) -> bool:
        # Called after every env.step(); returning False stops training
        return not self._stop_now

class RewardComponentLogger(BaseCallback):
    """
    Logs the avg of each reward component over one PPO iteration.
    Assumes RewardShapingWrapper has injected weighted terms into info[name].
    """
    def __init__(self, reward_names, verbose=0):
        super().__init__(verbose)
        # these are the keys you want to log, e.g. ["reach_reward","grasp_reward",…]
        self.reward_names = reward_names
        # accumulators
        self.sums  = {("reward_"+n): 0.0 for n in reward_names}
        self.count = 0

    def _on_step(self) -> bool:
        # called at every env.step(); collect infos from each parallel env
        infos = self.locals.get("infos", [])
        for info in infos:
            # print("info: ",  info)
            for name in self.reward_names:
                val = info.get("reward_"+name)
                if val is not None:
                    self.sums["reward_"+name] += val
            self.count += 1
        return True

    def _on_rollout_end(self) -> None:
        # called once per PPO iteration (end of rollout)
        for name in self.reward_names:
            if self.count > 0:
                avg = self.sums["reward_"+name] / self.count
                # avg = self.sums["reward_"+name]

                self.logger.record(f"reward/{"reward_"+name}", float(avg))
        # reset for next iteration
        self.sums  = {("reward_"+n): 0.0 for n in self.reward_names}
        self.count = 0

class AdaptiveLrCallback(BaseCallback):
    """
    IsaacSim-style “schedule='adaptive'” learning-rate adjustment based on KL.
    If kl > desired_kl*2   → lr ← max(min_lr, lr / 1.5)
    If kl < desired_kl/2   → lr ← min(max_lr, lr * 1.5)
    """
    def __init__(
        self,
        desired_kl: float,
        min_lr: float = 1e-5,
        max_lr: float = 1e-2,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.desired_kl = desired_kl
        self.min_lr     = min_lr
        self.max_lr     = max_lr

    def _on_step(self) -> bool:
        # Must return True to keep training going
        return True
    
    def _on_rollout_end(self) -> None:
        # read SB3’s approximate KL from the logger
        kl = self.model.logger.name_to_value.get("rollout/approx_kl")
        if kl is None or self.desired_kl is None:
            return

        # grab current lr (we assume a single optimizer on the policy)
        opt = self.model.policy.optimizer
        current_lr = opt.param_groups[0]["lr"]

        # IsaacSim logic
        if kl > self.desired_kl * 2.0:
            new_lr = max(self.min_lr, current_lr / 1.5)
        elif kl < self.desired_kl / 2.0 and kl > 0.0:
            new_lr = min(self.max_lr, current_lr * 1.5)
        else:
            return  # within tolerance, do nothing

        # apply to optimizer
        for g in opt.param_groups:
            g["lr"] = new_lr
        # keep SB3’s bookkeeping in sync
        self.model.learning_rate = new_lr

        if self.verbose:
            print(f"[AdaptiveLR] kl={kl:.4f}  "
                  f"lr: {current_lr:.6f} → {new_lr:.6f}")
        # log to TensorBoard under "train/learning_rate"
        self.logger.record("train/learning_rate", new_lr)