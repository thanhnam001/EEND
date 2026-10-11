#!/usr/bin/env python3

# Drop-in replacement for KaldiDiarizationDataset that reads mel
# spectrograms precomputed by the DiaPer-side precompute_features.py, and
# applies splice / subsample / top-N speaker selection at load time.
#
# Why this exists: this repo only knows how to read a Kaldi data dir and
# compute features from wav.scp. For the MSDWild/RAMC comparison the
# simulated-conversation corpora are only available as precomputed mel
# caches, not as audio, so there is no wav.scp to read. The features in
# those caches are produced by the same log-mel code as this repo's
# common_utils/features.transform(), and are stored *before* splice and
# subsample, so everything after that point is done here exactly as
# KaldiDiarizationDataset.__getitem__ does it.
#
# Cache contract (written by precompute_features.py):
#   meta.pkl                 -> dict with chunk_indices, feature_dim,
#                               input_transform, optional storage_format
#                               and chunk_locations
#   {i:08d}.pkl              -> per-chunk dict, storage_format 'perfile'
#   shard_NNNNN.bin          -> packed chunks, storage_format 'shard'
# each chunk being {'Y', 'T', 'rec', 'st', 'ed', 'speaker_ids'} where Y is
# the post-transform mel spectrogram.

import logging
import os
import pickle
from typing import Tuple

import numpy as np
import torch

import common_utils.features as features
# Side-effect import: lets pickle.load() read chunks written under
# numpy>=2.0 from a process running numpy<2.0. No-op otherwise.
import common_utils.numpy2_pickle_compat  # noqa: F401


# DataLoader workers are separate OS processes. A file handle opened
# before they fork shares one seek position across all of them, so
# concurrent seek+read would race. Open shard handles lazily and cache
# them per (path, pid) so a forked child never reuses a parent's
# shared-offset handle.
_shard_file_cache: dict = {}


def _get_shard_file(path: str):
    key = (path, os.getpid())
    f = _shard_file_cache.get(key)
    if f is None:
        f = open(path, 'rb')
        _shard_file_cache[key] = f
    return f


class PrecomputedKaldiDiarizationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        precomputed_dir: str,
        context_size: int,
        n_speakers: int,
        subsampling: int,
        feature_dim: int = None,
        input_transform: str = None,
        dtype: type = np.float32,
    ):
        self.precomputed_dir = precomputed_dir
        self.context_size = context_size
        self.n_speakers = n_speakers
        self.subsampling = subsampling
        self.dtype = dtype

        with open(os.path.join(precomputed_dir, 'meta.pkl'), 'rb') as f:
            self.meta = pickle.load(f)
        self.chunk_indices = self.meta['chunk_indices']
        # Absent in caches written before storage_format existed; those
        # are all 'perfile'.
        self.storage_format = self.meta.get('storage_format', 'perfile')
        if self.storage_format == 'shard':
            self.chunk_locations = self.meta['chunk_locations']

        # The cache fixes the feature pipeline up to transform(). Fail loudly
        # on a mismatch rather than training on features that do not match
        # the config, which would be near-impossible to spot later.
        for name, want in (('feature_dim', feature_dim),
                           ('input_transform', input_transform)):
            have = self.meta.get(name)
            if want is not None and have is not None and have != want:
                raise ValueError(
                    f"precomputed cache {precomputed_dir} was built with "
                    f"{name}={have!r} but this run is configured for "
                    f"{name}={want!r}; they must match")

        logging.info(
            f"Loaded precomputed dataset from {precomputed_dir}: "
            f"#chunks: {len(self.chunk_indices)} "
            f"(feature_dim={self.meta.get('feature_dim')}, "
            f"input_transform={self.meta.get('input_transform')}, "
            f"chunk_size={self.meta.get('chunk_size')} raw frames, "
            f"storage_format={self.storage_format})")

        self.saved = None  # reused when a chunk comes back empty

    def __len__(self) -> int:
        return len(self.chunk_indices)

    def _load_chunk(self, i: int):
        if self.storage_format == 'shard':
            shard_name, offset, length = self.chunk_locations[i]
            f = _get_shard_file(os.path.join(self.precomputed_dir, shard_name))
            f.seek(offset)
            d = pickle.loads(f.read(length))
        else:
            with open(os.path.join(self.precomputed_dir,
                                   f"{i:08d}.pkl"), 'rb') as f:
                d = pickle.load(f)
        return d['Y'], d['T'], d['rec']

    def __getitem__(self, i: int) -> Tuple[np.ndarray, np.ndarray, str]:
        Y, T, rec = self._load_chunk(i)

        if Y.shape[0] == 0:
            logging.warning(f"{rec} chunk {i} is empty: {Y.shape}, "
                            "replacing with previous sequence")
            Y, T, rec = self.saved
        else:
            self.saved = (Y, T, rec)

        # Identical to KaldiDiarizationDataset.__getitem__ from here on;
        # the cache stops right after features.transform().
        Y_spliced = features.splice(Y, self.context_size)
        Y_ss, T_ss = features.subsample(Y_spliced, T, self.subsampling)

        # If the sample contains more than "self.n_speakers" speakers,
        #  extract top-(self.n_speakers) speakers
        if self.n_speakers and T_ss.shape[1] > self.n_speakers:
            selected_spkrs = np.argsort(
                T_ss.sum(axis=0))[::-1][:self.n_speakers]
            T_ss = T_ss[:, selected_spkrs]

        return torch.from_numpy(np.copy(Y_ss).astype(self.dtype)), \
            torch.from_numpy(np.copy(T_ss)), rec
