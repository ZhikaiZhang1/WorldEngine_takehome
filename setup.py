from setuptools import setup, find_packages

setup(
    name="we_sim",
    version="0.1.0",
    description="Simulation environments for World Engine",
    author="World Engine Team",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "gymnasium",
        "mujoco",
        "mediapy",
        "tqdm",
        "mcap",
    ],
    python_requires=">=3.8",
    include_package_data=True,
    package_data={
        "we_sim": ["assets/**/*"],
    },
    entry_points={
        "console_scripts": [
            "render-mcap=we_sim.scripts.render_mcap:main",
            "run-block-pickup=we_sim.scripts.run_block_pickup:main",
        ],
    },
)
