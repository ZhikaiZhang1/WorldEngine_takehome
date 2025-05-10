"""
Simulation environments for the World Engine.
"""

from .base_env import WeEnv
from .mjx_envs.dual_piper_block_pickup_env import DualPiperBlockPickupEnv

__all__ = ["WeEnv", "DualPiperBlockPickupEnv", "get_env"]


def get_env(env_name: str, **env_kwargs) -> WeEnv:
    """
    Get an environment by name.

    Args:
        env_name: The name of the environment to get.

    Returns:
        The environment.

    Raises:
        ValueError: If the environment is not found.
    """
    if env_name == "dual_piper_block_pickup":
        return DualPiperBlockPickupEnv(**env_kwargs)
    else:
        raise ValueError(f"Environment {env_name} not found")
