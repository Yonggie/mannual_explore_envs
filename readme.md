Simplest scripts for mannually explore the simulation of Gibson, HM3D and MP3D!

Just make sure you have a minitor or X server etc. available!
# Environment
3 example envs options at ``envs/xx.glb``.
One glb is one env, you can also use your own glb file!
# requirements
```
# We require python>=3.9 and cmake>=3.10
conda create -n habitat python=3.9 cmake=3.14.0
conda activate habitat

conda install habitat-sim withbullet -c conda-forge -c aihabitat
```

PyTorch installation,  please refer to [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

# Run
```
python mannual_llm_explore.py
```

# Expected
