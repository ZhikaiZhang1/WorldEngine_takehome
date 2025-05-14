import torch.nn as nn
# ─── PPO / SB3 CONFIG ────────────────────────────────────────────────────────
PPO_PARAMS = {
    "policy": "MlpPolicy",
    "learning_rate": 1e-4,#3e-4,
    # "n_steps": 2048,
    # "n_steps": 64*8,

    "n_steps": 2048*16,
    # "n_steps": 4096,

    # "n_steps": 98304,
    # "n_steps": 200*16,


    "batch_size": 1024,#32,#192,
    # "batch_size": 800,

    # "batch_size": 1536,
    # "batch_size":24576,

    # "gamma": 0.98,#0.99,
    "gamma": 0.99,#0.99,

    "gae_lambda": 0.95,
    # "gae_lambda": 0.99,
    # "ent_coef": 0.006,#0.0,
    "ent_coef": 0.004,#1e-3,#0.0,

    "clip_range": 0.2,
    "vf_coef": 1.0,#1.0,#0.5,
    # "vf_coef": 0.5,

    "max_grad_norm": 1.0,#0.5,
    "n_epochs": 8,#15,#5,#10,
    # explicit policy_kwargs with net_arch and activation
    "policy_kwargs": {
        "net_arch": dict(pi=[256, 128, 64], vf=[256, 128, 64]),
        "activation_fn": nn.ELU,
    },
    # "policy_kwargs": {
    #     "net_arch": dict(pi=[512, 256,128], vf=[512, 256,128]),
    #     "activation_fn": nn.ReLU,
    # },
    "tensorboard_log": "logs/ppo_teacher",
    "verbose": 1,
    "target_kl":0.01,
    # "target_kl":0.02,

}

SAC_PARAMS = {
    "policy":          "MlpPolicy",
    "policy_kwargs": {
        "net_arch":       [256, 128,  64],
        "activation_fn":  nn.ELU,
    },
    # "policy_kwargs": {
    #     "net_arch":       [512, 256, 128],
    #     "activation_fn":  nn.ELU,
    # },

    # Optimization hyper-parameters
    "learning_rate":          3e-4,
    "buffer_size":            100_000,   # this is the replay buffer
    "learning_starts":        1_000,     # no updates until this many env steps
    "batch_size":             800,       # samples per gradient step
    "tau":                    0.005,     # smoothing tau
    "gamma":                  0.99,      

    # When/how often to update:
    "train_freq":             1,         # 1 update per env step
    "gradient_steps":         1,         # 1 minibatch each time

    # Entropy coefficient (temperature)  
    "ent_coef":               "auto",    # automatic α tuning

    # How often to update target networks
    "target_update_interval": 1,         # every gradient step

    # Logging & misc
    "tensorboard_log":        "logs/sac_teacher",
    "verbose":                1,
    "device":                 "cuda:0",  # or "cpu"
}

# ─── ENVIRONMENT SETTINGS ────────────────────────────────────────────────────
ENV_PARAMS = {
    "env_name": "dual_piper_block_pickup",
    "env_kwargs": {
        "render_mode": None,
        "max_episode_steps": 500,#250,
        "camera_view": "top-front",
        "randomize_block_positions": True,
        # used by RewardShapingWrapper
        # "reward_type": "custom",
    }
}

# ─── REWARD WEIGHTS ───────────────────────────────────────────────────────────
# Only weights > 0 are applied dynamically via RewardShapingWrapper
REWARD_WEIGHTS = {
    # "reach": 2.0,
    # "distance":1.0,
    # "wrong_arm_panelty": 0.1,
    "reach_inv_sq": 2.0,
    # "gaussian_reach": 2.0,
    # "reach_stage":100.0,
    "pick_stage":1.0,
    "place_success":1000,
    # "place_stage":5.0,
    # "drop_panelty_stage":1000.0,
    # "grasp_slip": 5.0,
    # "grasp_close": 2.0,
    # "lift": 15.0,
    "transport": 5.0,
    # "transport_putdown": 5.0,
    # "place_success": 10.0,
    # "place_accuracy": 3.0,
    # "action_penalty": 1e-4,
    # "closest_arm": 0.5,
    # "joint_vel_panelty":5e-6,
    # "far_arm_penalty": 0.1,
    # "time_penalty"     : 0.1,#0.1,#0.01,

}

# ─── TRAINING / CALLBACKS ─────────────────────────────────────────────────────
TRAINING = {
    "total_timesteps": 1_000_000,
    "total_iter": 50000,

    "eval_freq": 1000,
    "eval_episodes": 10,
    "save_freq": 50,
    "save_freq_iters":50,
    "save_path": "logs/ppo_teacher",
}

# ─── OTHER PARAMS ─────────────────────────────────────────────────────
OTHER_PARAMS = {
    "reach_std":0.1,#0.5,#0.1,
    "act_penalty_std":0.1,#0.5,#0.1,

    "close_thresh": 0.01,
    "reach_thresh": 0.05,
    
    "reach_inv_sq_thresh": 0.1,
    "minimal_height": 0.07,
    "putdown_thresh":0.05,
    "transport_std":0.3,
    "transport_putdown_std":0.05,
    "grasp_scale": 50,
    "maxPlacingDist":0.1,
    # "lin2tanh_connect":0.1,
    "lin2tanh_connect":0.1,
    "actiongrip_max": 0.04,
}
