World_engine_sim (Test version)

## Install

1. Use pip:
```
pip install -e .
```

2. Use UV (experimental)
```
uv sync
```
****
## Run

```python
python scripts/run_block_pickup.py --record-video
```

If you are using Mac, 
```
mjpython scripts/run_block_pickup.py
```

## FIXME:

1. We may need to manually draw the correct texture color.

## Trouble-shooting
To run the system on headless server.
```
xvfb-run python scripts/run_block_pickup.py
```

You may need to install `sudo apt-get install xvfb`.

Or set the Opengl end:
```
export MUJOCO_GL="egl"
```