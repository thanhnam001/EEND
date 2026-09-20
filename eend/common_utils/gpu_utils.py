#!/usr/bin/env python3

# Copyright 2022 Brno University of Technology (author: Federico Landini)
# Licensed under the MIT license.

# LOCAL PORT PATCH (2026-09-20, custom-diaper comparison setup):
# safe_gpu imports `fcntl`, which does not exist on Windows. safe_gpu only
# reserves a GPU to avoid collisions between concurrent jobs; it is not part
# of the method. Make it optional so the repo runs on this machine, and fall
# back to plain CUDA device selection. Upstream file is otherwise untouched.
try:
    from safe_gpu import safe_gpu
    _SAFE_GPU = True
except (ImportError, ModuleNotFoundError):
    safe_gpu = None
    _SAFE_GPU = False


def use_single_gpu(gpu_id):
    if _SAFE_GPU:
        gpu_owner = safe_gpu.GPUOwner(nb_gpus=1)
        return gpu_owner.devices_taken[0]
    return int(gpu_id) if gpu_id is not None else 0


def use_gpus(gpus_qty):
    if _SAFE_GPU:
        return safe_gpu.GPUOwner(nb_gpus=gpus_qty)
    return None
