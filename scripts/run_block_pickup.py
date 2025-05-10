#!/usr/bin/env python
"""
Command-line script for running the block pickup environment.
"""

import argparse
import numpy as np
import os
import mediapy as media
from we_sim.envs import get_env
import re
from stable_baselines3 import PPO

def flatten_privileged(obs: dict) -> np.ndarray:
    """
    Given the full obs dict from DualPiperBlockPickupEnv,
    extract and flatten exactly the same vector the teacher policy was trained on.
    """
    return np.concatenate([
        obs["joint_positions"],
        obs["joint_velocities"],
        obs["block_positions"],
        obs["target_positions"],
        obs["gripper_positions"],
    ])

def find_latest_model(model_dir: str) -> str:
    """
    Scan `model_dir` for files matching "*_<n>_steps.zip" and return
    the one with the highest <n>.
    """
    pattern = re.compile(r'.*_(\d+)_steps(?:_.*)?\.zip$')
    candidates = []
    for fname in os.listdir(model_dir):
        match = pattern.match(fname)
        if match:
            steps = int(match.group(1))
            candidates.append((steps, fname))
    if not candidates:
        raise FileNotFoundError(f"No model files matching '*_<n>_steps.zip' in {model_dir}")
    # pick the one with the largest step count
    latest = max(candidates, key=lambda x: x[0])[1]
    return os.path.join(model_dir, latest)

def main():
    """
    Main function for command-line usage.
    """
    parser = argparse.ArgumentParser(description="Run the block pickup environment")
    parser.add_argument("--model-path",  type=str, required=True,help="Path to the .zip SB3 PPO model file")
    parser.add_argument("--render-mode", choices=["human", "rgb_array", "mujoco_gui"], default="mujoco_gui", help="Rendering mode")
    parser.add_argument("--camera", choices=["front", "top", "top-front"], default="top-front", help="Camera view perspective")
    parser.add_argument("--steps", type=int, default=200, help="Number of steps to run")
    parser.add_argument("--reward", choices=["sparse", "dense"], default="sparse", help="Reward type")
    parser.add_argument("--still", action="store_true", help="Run in still mode")
    parser.add_argument("--record-video", action="store_true", help="Record video from all cameras")
    parser.add_argument("--output-dir", type=str, default="camera_recordings", help="Directory to save video recordings")
    parser.add_argument("--fps", type=int, default=30, help="Frame rate for recorded videos")
    args = parser.parse_args()

    # Create output directory if recording video
    if args.record_video:
        os.makedirs(args.output_dir, exist_ok=True)
        video_buffers = {}
    basepath = "../../logs/ppo_teacher"
    topdir = os.path.join(basepath, args.model_path)
    model_zip = find_latest_model(topdir)
    model = PPO.load(model_zip)
    env = get_env("dual_piper_block_pickup", render_mode=args.render_mode)
    observation, info = env.reset()

    for step in range(args.steps):
        print(f"Step {step} of {args.steps}")
        if args.still:
            action = np.zeros(env.action_space.shape)
        else:
            # action = env.action_space.sample()  # Random action
            vec_obs = flatten_privileged(observation)
            # model.predict returns (action, state); we ignore recurrent state
            action, _states = model.predict(vec_obs, deterministic=True)
            # print("action is:", action)
        observation, reward, terminated, truncated, info = env.step(action)

        # If recording video, store frames from all cameras
        if args.record_video and "color_images" in observation:
            for cam_name, frame in observation["color_images"].items():
                if cam_name not in video_buffers:
                    video_buffers[cam_name] = []
                video_buffers[cam_name].append(frame)

        if terminated or truncated:
            observation, info = env.reset()

    # Save videos if recording was enabled
    if args.record_video and video_buffers:
        print(f"Saving camera recordings to {args.output_dir}...")
        for cam_name, frames in video_buffers.items():
            if frames:  # Check if we have any frames
                frames = np.stack(frames)  # Stack frames into a single array
                output_path = os.path.join(args.output_dir, f"{cam_name}.mp4")
                # Save as MP4 using mediapy
                media.write_video(output_path, frames, fps=args.fps)
                print(f"Saved {cam_name} recording to {output_path}")

    env.close()

if __name__ == "__main__":
    main()
