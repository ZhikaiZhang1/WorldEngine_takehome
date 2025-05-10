import torch.nn as nn
# ─── PPO / SB3 CONFIG ────────────────────────────────────────────────────────
PPO_PARAMS = {
    # Policy type: use MlpPolicy (built‑in) with custom net_arch
    "policy": "MlpPolicy",
    "learning_rate": 1e-4,#3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "gamma": 0.98,#0.99,
    "gae_lambda": 0.95,
    "ent_coef": 0.006,#0.0,
    "clip_range": 0.2,
    "vf_coef": 1.0,#0.5,
    "max_grad_norm": 1.0,#0.5,
    "n_epochs": 5,#10,
    # explicit policy_kwargs with net_arch and activation
    "policy_kwargs": {
        "net_arch": dict(pi=[256, 128, 64], vf=[256, 128, 64]),
        "activation_fn": nn.ELU,
    },
    "tensorboard_log": "logs/ppo_teacher",
    "verbose": 1,
}

# ─── ENVIRONMENT SETTINGS ────────────────────────────────────────────────────
ENV_PARAMS = {
    "env_name": "dual_piper_block_pickup",
    "env_kwargs": {
        "render_mode": None,
        "max_episode_steps": 500,
        "camera_view": "top-front",
        "randomize_block_positions": True,
        # used by RewardShapingWrapper
        # "reward_type": "custom",
    }
}

# ─── REWARD WEIGHTS ───────────────────────────────────────────────────────────
# Only weights > 0 are applied dynamically via RewardShapingWrapper
REWARD_WEIGHTS = {
    "reach": 1.0,
    # "grasp_slip": 5.0,
    # "grasp_close": 2.0,
    "lift": 10.0,
    "transport": 10.0,
    "transport_putdown": 5.0,
    # "place_success": 10.0,
    # "place_accuracy": 3.0,
    "action_penalty": 1e-4,
    # "closest_arm": 0.5,
    "joint_vel_panelty":1e-4,
    # "far_arm_penalty": 0.1,
}

# ─── TRAINING / CALLBACKS ─────────────────────────────────────────────────────
TRAINING = {
    "total_timesteps": 1_000_000,
    "eval_freq": 50_000,
    "eval_episodes": 10,
    "save_freq": 200,
    "save_path": "logs/ppo_teacher",
}

# ─── OTHER PARAMS ─────────────────────────────────────────────────────
OTHER_PARAMS = {
    "reach_std":0.1,
    "close_thresh": 0.01,
    "minimal_height": 0.04,
    "putdown_thresh":0.01,
    "transport_std":0.3,
    "transport_putdown_std":0.05,

}
