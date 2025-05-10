
        
import os

os.environ["MUJOCO_GL"] = "osmesa"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
import argparse
import numpy as np
import mediapy as media

from teacher_config import PPO_PARAMS, ENV_PARAMS, TRAINING
from env_robot import PrivilegedObsWrapper, RewardShapingWrapper, make_callbacks, EpisodeRewardCallback
# from .mjx_envs.dual_piper_block_pickup_env import DualPiperBlockPickupEnv

from we_sim.envs import get_env


args = None
# Environment constructor
def make_env():
    # Copy env_kwargs and extract reward_type
    env_kwargs = ENV_PARAMS["env_kwargs"].copy()
    # reward_type = env_kwargs.pop("reward_type", "dense")

    # Directly instantiate your custom environment
    env = get_env("dual_piper_block_pickup", render_mode=args.render_mode)

    # env = DualPiperBlockPickupEnv(**env_kwargs)

    # Apply wrappers for custom rewards and privileged obs
    env = RewardShapingWrapper(env)
    env = PrivilegedObsWrapper(env)
    return env

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the block pickup environment")
    parser.add_argument("--render-mode", choices=["human", "rgb_array", "mujoco_gui"], default="rgb_array", help="Rendering mode")

    # parser.add_argument("--render-mode", choices=["human", "rgb_array", "mujoco_gui"], default="mujoco_gui", help="Rendering mode")
    parser.add_argument("--camera", choices=["front", "top", "top-front"], default="top-front", help="Camera view perspective")
    parser.add_argument("--steps", type=int, default=200, help="Number of steps to run")
    parser.add_argument("--reward", choices=["sparse", "dense"], default="sparse", help="Reward type")
    parser.add_argument("--still", action="store_true", help="Run in still mode")
    parser.add_argument("--record-video", action="store_true", help="Record video from all cameras")
    parser.add_argument("--output-dir", type=str, default="camera_recordings", help="Directory to save video recordings")
    parser.add_argument("--fps", type=int, default=30, help="Frame rate for recorded videos")
    parser.add_argument("--headless", action="store_true", default=False,help="headless mode")
    parser.add_argument("--device", type=str, default="cpu", help="device to run on")



    args = parser.parse_args()


    # Ensure directories exist
    os.makedirs(PPO_PARAMS["tensorboard_log"], exist_ok=True)
    # os.makedirs(TRAINING["save_path"], exist_ok=True)

    # Vectorized environments
    train_env = DummyVecEnv([make_env])
    train_env = VecMonitor(train_env, PPO_PARAMS["tensorboard_log"])
    # eval_env  = DummyVecEnv([make_env])
    eval_env  = VecMonitor(DummyVecEnv([make_env]), PPO_PARAMS["tensorboard_log"])
    




    # 1) Instantiate your env directly—no wrappers
    # raw_env = DualPiperBlockPickupEnv(**ENV_PARAMS["env_kwargs"])
    raw_env = get_env("dual_piper_block_pickup", render_mode=args.render_mode)


    # 2) Read out the MuJoCo control timestep
    dt = raw_env.sim.model.opt.timestep
    max_steps = ENV_PARAMS["env_kwargs"]["max_episode_steps"]
    print(f"Control timestep (dt) = {dt:.4f} s")
    print(f"Max episode steps = {max_steps} → episode length ≃ {dt * max_steps:.2f} s")

    # 3) Always close when you’re done
    raw_env.close()



    
    # Instantiate PPO model
    model = PPO(
        policy=PPO_PARAMS["policy"],
        env=train_env,
        learning_rate=PPO_PARAMS["learning_rate"],
        n_steps=PPO_PARAMS["n_steps"],
        batch_size=PPO_PARAMS["batch_size"],
        gamma=PPO_PARAMS["gamma"],
        gae_lambda=PPO_PARAMS["gae_lambda"],
        ent_coef=PPO_PARAMS["ent_coef"],
        clip_range=PPO_PARAMS["clip_range"],
        vf_coef=PPO_PARAMS["vf_coef"],
        max_grad_norm=PPO_PARAMS["max_grad_norm"],
        n_epochs=PPO_PARAMS["n_epochs"],
        policy_kwargs=PPO_PARAMS["policy_kwargs"],
        tensorboard_log=PPO_PARAMS["tensorboard_log"],
        verbose=PPO_PARAMS["verbose"],
    )

    # Callbacks for eval and checkpoint
    savepath = model.logger.dir
    callbacks = make_callbacks(eval_env, TRAINING, savepath)
    # model.save(os.path.join(model.logger.dir, "ppo_teacher_final"))

    callbacks.insert(0, EpisodeRewardCallback())
    # dt = train_env.sim.model.opt.timestep
    # print("env dt is: ", dt)
    
    # Train
    model.learn(
        total_timesteps=TRAINING["total_timesteps"],
        callback=callbacks,
        log_interval    = 1, 
    )
    # import ipdb;ipdb.set_trace()


    # Save final policy
    # model.save(os.path.join(TRAINING["save_path"], "ppo_teacher_final"))
    model.save(os.path.join(model.logger.dir, "ppo_teacher_final"))