#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pints.io

iid_noise_sigma = 25  # pA
file_chain_A = 'out/mcmc-model_A/model_A-sinewave-chain_0.csv'
file_chain_B = 'out/mcmc-model_B/model_B-sinewave-chain_0.csv'

chain_A = pints.io.load_samples(file_chain_A, n=None)
chain_B = pints.io.load_samples(file_chain_B, n=None)

sigma_A = chain_A[-50000:, -1]
sigma_B = chain_B[-50000:, -1]

n_bins = np.arange(20, 55, 0.05)

plt.figure(figsize=(6, 3))
plt.hist(sigma_A, bins=n_bins, alpha=0.75, density=True,
        label='Model A (mean=%.2f pA)' % np.mean(sigma_A))
plt.hist(sigma_B, bins=n_bins, alpha=0.75, density=True,
        label='Model B (mean=%.2f pA)' % np.mean(sigma_B))
plt.axvline(iid_noise_sigma, ls='--', c='#7f7f7f',
        label=r'True $\sigma$ in data')
plt.xlim((24, 51))
plt.xlabel(r'$\sigma$ (pA)')
plt.ylabel('Posterior')
plt.legend()
plt.tight_layout()
plt.savefig('fig/iid-sigma.pdf', format='pdf', bbox_inches='tight')
plt.close()
