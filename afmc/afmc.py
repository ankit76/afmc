import os
#os.environ['JAX_ENABLE_X64'] = 'True'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
from itertools import product
import numpy as np
#from scipy.fft import fft, ifft, ifftshift, ifftn
from jax.numpy.fft import fft, ifft, ifftshift, ifftn, fftshift, fftn
from jax import lax, jit, value_and_grad, random, vmap,numpy as jnp
import jax
from functools import partial
import matplotlib.pyplot as plt

def blocking_analysis(weights, energies, neql = 0, printQ=False, writeBlockedQ=False):
  nSamples = weights.shape[0] - neql
  weights = weights[neql:]
  energies = energies[neql:]
  weightedEnergies = np.multiply(weights, energies)
  meanEnergy = weightedEnergies.sum() / weights.sum()
  if printQ:
    print(f'#\n# Mean energy: {meanEnergy:.8e}')
    print('# Block size    # of blocks         Mean                Error')
  blockSizes = np.array([ 1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500, 1000, 10000 ])
  prevError = 0.
  plateauError = None
  for i in blockSizes[blockSizes < nSamples/2.]:
    nBlocks = nSamples//i
    blockedWeights = np.zeros(nBlocks)
    blockedEnergies = np.zeros(nBlocks)
    for j in range(nBlocks):
      blockedWeights[j] = weights[j*i:(j+1)*i].sum()
      blockedEnergies[j] = weightedEnergies[j*i:(j+1)*i].sum() / blockedWeights[j]
    v1 = blockedWeights.sum()
    v2 = (blockedWeights**2).sum()
    mean = np.multiply(blockedWeights, blockedEnergies).sum() / v1
    error = (np.multiply(blockedWeights, (blockedEnergies - mean)**2).sum() / (v1 - v2 / v1) / (nBlocks - 1))**0.5
    if writeBlockedQ:
      np.savetxt(f'samples_blocked_{i}.dat', np.stack((blockedWeights, blockedEnergies)).T)
    if printQ:
      print(f'  {i:5d}           {nBlocks:6d}       {mean:.8e}       {error:.6e}')
    if error < 1.05 * prevError and plateauError is None:
      plateauError = max(error, prevError)
    prevError = error

  if printQ:
    if plateauError is not None:
      print(f'# Stocahstic error estimate: {plateauError:.6e}\n#')

  return meanEnergy, plateauError

@jit
def symmetrize_1d(fields):
  n_fields_1d = fields.shape[0]
  fields = fields.at[:n_fields_1d//2].set(jnp.conj(jnp.flip(fields[n_fields_1d//2+1:])))
  return fields

@jit
def symmetrize_2d(fields):
  n_fields_1d = fields.shape[0]
  fields = fields.at[:,:n_fields_1d//2].set(jnp.conj(jnp.flip(fields[:,n_fields_1d//2+1:])))
  fields = fields.at[:n_fields_1d//2,n_fields_1d//2].set(jnp.conj(jnp.flip(fields[n_fields_1d//2+1:,n_fields_1d//2])))
  return fields

@jit
def symmetrize(fields):
  n_fields_1d = fields.shape[0]
  fields = fields.at[:,:,:n_fields_1d//2].set(jnp.conj(jnp.flip(fields[:,:,n_fields_1d//2+1:])))
  fields = fields.at[:,:n_fields_1d//2,n_fields_1d//2].set(jnp.conj(jnp.flip(fields[:,n_fields_1d//2+1:,n_fields_1d//2])))
  fields = fields.at[:n_fields_1d//2,n_fields_1d//2,n_fields_1d//2].set(jnp.conj(jnp.flip(fields[n_fields_1d//2+1:,n_fields_1d//2,n_fields_1d//2])))
  return fields


@jit
def calc_denom_sample(fields_g, sign_mask, n_particles):
  fields_r = ifftn(ifftshift(fields_g), norm="forward")
  #jax.debug.print('fields_g:\n{}', fields_g)
  #jax.debug.print('fields_r:\n{}', jnp.real(fftshift(fields_r)))
  #jax.debug.print('int fields_r:\n{}', jnp.sum(jnp.exp(-1.j * fields_r) * dV) / dV / jnp.size(fields_g))
  #jax.debug.print('denom_sample: {}', (jnp.sum(jnp.exp(-fields_r * 1.j) / jnp.size(fields_g)))**N)

  #return jnp.log(jnp.sum(jnp.cos(-fields_r) / jnp.size(fields_g))) * N
  return jnp.log(jnp.sum(jnp.exp(fields_r * sign_mask) / jnp.size(fields_g))) * n_particles
  #return (jnp.sum(jnp.exp(-fields_r * 1.j) / jnp.size(fields_g)))**N

@jit
def calc_vol_int(fields_g):
  fields_r = ifftn(ifftshift(fields_g), norm="forward")
  i_g = fftshift(fftn(jnp.exp(-1.j * fields_r)) / jnp.size(fields_g))
  return i_g


@jit
def calc_pot_ene(fields_g, phi_g):
  return jnp.sum(jnp.abs(fields_g)**2 / jnp.abs(phi_g))

@jit
def direct_sampling(phi_g, beta, L, N, mask, sign_mask, n_samples_proxy, seed=0):
  nkpts_1d = phi_g.shape[0]
  nkpts_fields = jnp.count_nonzero(mask)
  dL = L / nkpts_1d
  dV = dL**3
  phi_0 = phi_g[nkpts_1d//2, nkpts_1d//2, nkpts_1d//2]

  # carry: key
  def scanned_fun(carry, x):
    fields_g = 0.j * phi_g
    carry, subkey = random.split(carry)
    fields_g_r = (beta * jnp.abs(phi_g[:,:,nkpts_1d//2:]))**0.5 * random.normal(subkey, shape=(nkpts_1d, nkpts_1d, nkpts_1d//2+1))
    carry, subkey = random.split(carry)
    fields_g_i = (beta * jnp.abs(phi_g[:,:,nkpts_1d//2:]))**0.5 * random.normal(subkey, shape=(nkpts_1d, nkpts_1d, nkpts_1d//2+1))
    fields_g = fields_g.at[:,:,nkpts_1d//2:].set(fields_g_r)
    fields_g = fields_g.at[:,:,nkpts_1d//2:].add(1.j * fields_g_i)
    fields_g = mask * symmetrize(fields_g)
    fields_g = fields_g.at[nkpts_1d//2, nkpts_1d//2, nkpts_1d//2].set(0.j)
    denom_sample = calc_denom_sample(fields_g, sign_mask, N)
    i_g = calc_vol_int(fields_g)
    #corr_sample = i_g * jnp.flip(i_g)
    #corr_sample = corr_sample / corr_sample[nkpts_1d//2, nkpts_1d//2, nkpts_1d//2]
    #fields_r = ifftn(ifftshift(fields_g), norm="forward")
    #denom_sample = (jnp.sum(jnp.exp(-fields_r * 1.j)) * dV)**N
    pot_ene_sample = - calc_pot_ene(fields_g, phi_g) / 4 / beta**2
    return carry, (denom_sample, pot_ene_sample)

  key = random.PRNGKey(seed)
  n_samples = n_samples_proxy.shape[0]
  _, (denom_samples, pot_ene_samples) = lax.scan(scanned_fun, key, jnp.arange(n_samples))
  #jax.debug.print('denom_samples: {}', denom_samples)
  #jax.debug.print('pot_ene_samples: {}', pot_ene_samples)

  #avg_pot_ene = jnp.sum(denom_samples * pot_ene_samples) / jnp.sum(denom_samples)
  #avg_pot_ene += (N**2 * phi_0 / 2. - N / 2. + (nkpts_fields - 1) / 2 / beta)
  return denom_samples, pot_ene_samples
  #return avg_pot_ene / N, (denom_samples, pot_ene_samples)

if __name__ == "__main__":
  #for beta in [ 1000., 100., 10., 1., 0.1, 0.01 ]:
  nruns = 5
  pot_ene = np.zeros(nruns)
  for run in range(nruns):
    for beta in [ 0.1 ]:
      #beta = 0.001
      L = 16.**(1/3)
      rho = 1.
      N = 16

      nkpts = 7
      kpts_1d = np.array([2 * n * np.pi/L for n in range(-nkpts//2+1, nkpts//2+1)])
      kpts = np.array(list(product(kpts_1d, kpts_1d, kpts_1d))).reshape(nkpts, nkpts, nkpts, 3)
      phi_1d = np.array([np.exp(-(2 * n * np.pi/L)**2 / 4) * np.pi**0.5 / L for n in range(-nkpts//2+1, nkpts//2+1)])
      phi = np.einsum('i,j,k', phi_1d, phi_1d, phi_1d)

      e_cut = 30.
      g_cut = (e_cut)**0.5
      mask = np.zeros((nkpts, nkpts, nkpts))
      sign_mask = np.zeros((nkpts, nkpts, nkpts))+0.j      
      for i in range(nkpts):
        for j in range(nkpts):
          for k in range(nkpts):
            if np.linalg.norm(kpts[i,j,k]) < g_cut:
              mask[i,j,k] = 1.
            if phi[i,j,k] < 0:
              sign_mask[i,j,k] = 1.
            else:
              sign_mask[i,j,k] = -1.j

      phi_g = jnp.array(phi)
      mask = jnp.array(mask)
      n_samples = 10000
      n_samples_proxy = jnp.zeros((n_samples))

      denom_samples, pot_ene_samples = direct_sampling(phi_g, beta, L, N, mask, sign_mask, n_samples_proxy, seed=run)
      denom_samples = np.array(denom_samples)
      pot_ene_samples = np.array(pot_ene_samples)
      #print(f'denom_samples: {denom_samples}')
      n_large = n_samples
      indices = np.argsort(-np.real(denom_samples))
      sorted_denom = denom_samples[indices]
      sorted_denom = np.exp(sorted_denom - np.real(sorted_denom[0]))
      #print(f'sorted_denom: {sorted_denom}')
      sorted_pot = pot_ene_samples[indices]
      nkpts_1d = phi_g.shape[0]
      nkpts_fields = jnp.count_nonzero(mask)
      phi_0 = phi_g[nkpts_1d//2, nkpts_1d//2, nkpts_1d//2]
      avg_pot_ene = np.sum(sorted_denom[:n_large] * sorted_pot[:n_large]) / np.sum(sorted_denom[:n_large]) + (N**2 * phi_0 / 2. - N / 2. + (nkpts_fields - 1) / 2 / beta)
      #print(f'beta: {beta}, avg_pot_ene: {avg_pot_ene/N}')
      pot_ene[run] = np.real(avg_pot_ene / N)

      #avg_pot_ene, (denom_samples, pot_ene_samples) = direct_sampling(phi_g, beta, L, N, mask, n_samples_proxy, seed=3959)
      #print(f'avg pot ene: {avg_pot_ene}')
      #print(f'avg sign: {jnp.sum(denom_samples) / jnp.sum(jnp.abs(denom_samples))}')
      #print(f'avg int: {jnp.sum(-denom_samples * pot_ene_samples * 4 * beta**2) / jnp.sum(denom_samples)}')
      #print(f'beta nk: {-beta * (jnp.count_nonzero(mask)-1)}')

  np.savetxt('pot_ene_samples.dat', pot_ene)
  print(f'pot_ene: {np.mean(pot_ene)} +/- {np.std(pot_ene) / (nruns - 1)**0.5}')
  
