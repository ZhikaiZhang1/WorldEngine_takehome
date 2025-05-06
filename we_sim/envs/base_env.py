"""Base environment class for all World Engine environments."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, List, Any, Union
from we_sim.utils.async_util import AsyncUtil
import asyncio
import os
import enum


class Resolution(enum.Enum):
    """
    Standard video resolutions.
    """

    SD = (480, 640)
    HD = (720, 1280)
    UHD = (2160, 3840)


class WeEnv(gym.Env):
    """Base environment class for all World Engine environments.
    Supports asynchronous methods.
    """

    metadata = {"render_modes": ["human", "rgb_array", "mujoco_gui"], "render_fps": 60}
    image_keys = ["top_camera"]
    res = Resolution.HD
    h, w = res.value
    cameras = {}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        res_str: str = "HD",
        max_episode_steps: int = 500,
        num_dof: int = 0,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.num_dof = num_dof
        self.steps = 0
        self.res = Resolution[res_str]
        self.h, self.w = self.res.value
        self.cur_action = np.zeros(self.num_dof)
        self.read_write_mutex = asyncio.Lock()

    def _get_joint_state(self) -> np.ndarray:
        """Get the joint state of the environment. Syncronized version."""
        raise NotImplementedError

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Get the observation of the environment. Syncronized version."""
        raise NotImplementedError

    async def get_joint_state(self) -> np.ndarray:
        """Get the joint state of the environment. Asynchronous version."""
        async with self.read_write_mutex:
            return await AsyncUtil.run_blocking_as_async(self._get_joint_state)

    async def set_action(self, action: np.ndarray) -> None:
        """Set the action of the environment. Used in the asynchronous mode."""
        async with self.read_write_mutex:
            self.cur_action = action
            # Apply action
            await AsyncUtil.run_blocking_as_async(self.step)

    async def get_obs(self) -> Dict[str, np.ndarray]:
        """Get the observation of the environment. Asynchronous version."""
        async with self.read_write_mutex:
            return await AsyncUtil.run_blocking_as_async(self._get_obs)

    async def async_reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment. Asynchronous version."""
        async with self.read_write_mutex:
            return await AsyncUtil.run_blocking_as_async(self.reset, seed, options)
