#!/usr/bin/env python3
import numpy as np

dt = 0.2  # ms

# Stimulate at 1 Hz
stim1hz = []
for _ in range(5):
    stim1hz.append((0, 50))
    stim1hz.append((1, 1))  # stim duration use default
    stim1hz.append((0, 949))

stim1hz_times = np.arange(0, np.sum(np.asarray(stim1hz)[:, 1]), dt)

# Stimulate at 2 Hz
stim2hz = []
for _ in range(5):
    stim2hz.append((0, 50))
    stim2hz.append((1, 1))  # stim duration use default
    stim2hz.append((0, 449))

stim2hz_times = np.arange(0, np.sum(np.asarray(stim2hz)[:, 1]), dt)

# Random stimulus
np.random.seed(101)
randstim = [(0, 50)]
for r in np.random.uniform(100, 600, size=10):
    randstim.append((1, 1))  # stim duration use default
    randstim.append((0, r))
randstim.append((1, 1))  # stim duration use default
randstim.append((0, 500))

randstim_times = np.arange(0, np.sum(np.asarray(randstim)[:, 1]), dt)

# hERG block (0%, 25%, 50%, 75%, 100%)
from model import parameters
ikridx = parameters.index('ikr.s')

hergblock = np.ones((5, len(parameters)))
hergblock[:, ikridx] = 1. - np.array([0, 0.25, 0.5, 0.75, 1.0])
# print(hergblock * (2. * np.ones(len(parameters))))

stim1hz_hergblock = []
stim1hz_hergblock.append((0, 50))
stim1hz_hergblock.append((1, 1))  # stim duration use default
stim1hz_hergblock.append((0, 949))

stim1hz_hergblock_times = np.arange(0,
        np.sum(np.asarray(stim1hz_hergblock)[:, 1]), dt)

