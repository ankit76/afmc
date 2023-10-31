import os

os.environ["JAX_ENABLE_X64"] = "True"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import pickle
from functools import partial
from itertools import product
from typing import Any, Callable

import jax
import numpy as np
import stat_utils
from flax import struct
from jax import jit, lax
from jax import numpy as jnp
from jax import random, value_and_grad, vmap

# from scipy.fft import fft, ifft, ifftshift, ifftn
from jax.numpy.fft import fft, fftn, fftshift, ifft, ifftn, ifftshift
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


@struct.dataclass
class system:
    beta: float
    phi: Any = struct.field(pytree_node=False)
    grid: Any = struct.field(pytree_node=False)
    n_particles: int = struct.field(pytree_node=False)


@jit
def calc_fluctuation(fields_g, system_data):
    r_grid = system_data.grid.get_r_grid()
    unique_fields = fields_g[jnp.array(system_data.grid.unique_indices)]
    unique_k_points = jnp.array(system_data.grid.unique_k_points)
    fields_r = jnp.sum(
        vmap(
            lambda g, sigma_g: 2.0
            * sigma_g.real
            * jnp.cos(2.0 * jnp.pi * jnp.sum(g * r_grid, axis=1))
            - 2.0 * sigma_g.imag * jnp.sin(2.0 * jnp.pi * jnp.sum(g * r_grid, axis=1)),
            in_axes=(0, 0),
        )(unique_k_points, unique_fields),
        axis=0,
    )
    return jnp.sum(jnp.exp(-fields_r * 1.0j) / jnp.size(fields_r))


# @partial(jit, static_argnums=(2,))
@jit
def calc_pot_ene(fields_g, system_data):
    return (
        -jnp.sum(jnp.abs(fields_g) ** 2 / system_data.phi) / 2 / system_data.beta**2
    )


@jit
def calc_rdf(fields_g, system_data):
    r_grid = system_data.grid.get_r_grid()
    unique_fields = fields_g[jnp.array(system_data.grid.unique_indices)]
    unique_k_points = jnp.array(system_data.grid.unique_k_points)
    fields_r = jnp.sum(
        vmap(
            lambda g, sigma_g: 2.0
            * sigma_g.real
            * jnp.cos(2.0 * jnp.pi * jnp.sum(g * r_grid, axis=1))
            - 2.0 * sigma_g.imag * jnp.sin(2.0 * jnp.pi * jnp.sum(g * r_grid, axis=1)),
            in_axes=(0, 0),
        )(unique_k_points, unique_fields),
        axis=0,
    )
    eigr = vmap(lambda g: jnp.exp(1.0j * jnp.pi * 2.0 * jnp.sum(g * r_grid, axis=1)))(
        unique_k_points
    )
    i_g = (
        jnp.einsum("i,ji->j", jnp.exp(-fields_r * 1.0j), eigr)
        / jnp.size(fields_r)
        * jnp.einsum("i,ji->j", jnp.exp(-fields_r * 1.0j), jnp.conj(eigr))
        / jnp.size(fields_r)
        / (jnp.sum(jnp.exp(-fields_r * 1.0j) / jnp.size(fields_r))) ** 2
    )
    return i_g


# @partial(jit, static_argnums=(0,))
@jit
def initialize_fields(system_data):
    fields_data = {}
    fields_data["fields"] = 0.0j * system_data.phi
    fields_data["fluctuation"] = calc_fluctuation(fields_data["fields"], system_data)
    fields_data["energy"] = calc_pot_ene(fields_data["fields"], system_data)
    fields_data["phase"] = jnp.exp(
        1.0j
        * (
            system_data.n_particles
            * jnp.angle(fields_data["fluctuation"])
            % (2 * jnp.pi)
        )
    )
    fields_data["rdf"] = calc_rdf(fields_data["fields"], system_data)
    fields_data["n_accepted"] = 0
    return fields_data


# @partial(jit, static_argnums=(3,))
@jit
def metropolis_step(fields_data, move, random_num, system_data):
    fields = fields_data["fields"]
    fluctuation = fields_data["fluctuation"]
    new_fields = system_data.grid.symmetric_update(fields, move)
    new_fluctuation = calc_fluctuation(new_fields, system_data)
    # jax.debug.print('new_fluctuation: {}', new_fluctuation)
    log_exp_factor_ratio = (
        (-jnp.abs(new_fields[move[0]]) ** 2 + jnp.abs(fields[move[0]]) ** 2)
        / system_data.beta
        / system_data.phi[move[0]]
    )
    # jax.debug.print('log_exp_factor_ratio: {}', log_exp_factor_ratio)
    log_fluctuation_ratio = system_data.n_particles * jnp.log(
        jnp.abs(new_fluctuation) / jnp.abs(fluctuation)
    )
    # jax.debug.print('log_fluctuation_ratio: {}', log_fluctuation_ratio)
    log_b_ratio = log_exp_factor_ratio + log_fluctuation_ratio
    # jax.debug.print('log_b_ratio: {}', log_b_ratio)
    # jax.debug.print('fluc_ratio: {}', (jnp.abs(new_fluctuation) / jnp.abs(fluctuation))**5)
    fields_data["fields"] = lax.cond(
        log_b_ratio > jnp.log(random_num), lambda x: new_fields, lambda x: fields, 0
    )
    fields_data["fluctuation"] = lax.cond(
        log_b_ratio > jnp.log(random_num),
        lambda x: new_fluctuation,
        lambda x: fluctuation,
        0,
    )
    fields_data["energy"] = calc_pot_ene(fields_data["fields"], system_data)
    fields_data["rdf"] = calc_rdf(fields_data["fields"], system_data)
    fields_data["phase"] = jnp.exp(
        1.0j
        * (
            (system_data.n_particles * jnp.angle(fields_data["fluctuation"]))
            % (2 * jnp.pi)
        )
    )
    fields_data["n_accepted"] += log_b_ratio > jnp.log(random_num)
    return fields_data


# @partial(jit, static_argnums=(0,))
@jit
def sampling(system_data, random_data):
    random_numbers, random_proposals, random_indices = random_data

    def scanned_fun(carry, x):
        random_num = random_numbers[x]
        random_proposal = random_proposals[x]
        random_index = random_indices[x]
        move = (system_data.grid.get_grid_index(random_index), random_proposal)
        # jax.debug.print('\niter: {}', x)
        # jax.debug.print('random_index: {}', random_index)
        # jax.debug.print('random_proposal: {}', random_proposal)
        # jax.debug.print('random_num: {}', random_num)
        # jax.debug.print('move: {}', move)
        carry = metropolis_step(carry, move, random_num, system_data)
        # jax.debug.print('field_data:\n{}\n', carry)
        return carry, (
            carry["fields"],
            carry["energy"],
            carry["phase"],
            carry["fluctuation"],
            carry["rdf"],
        )

    fields_data = initialize_fields(system_data)
    fields_data, samples = lax.scan(
        scanned_fun, fields_data, jnp.arange(random_numbers.shape[0])
    )
    return samples, fields_data["n_accepted"] / random_numbers.shape[0]


def driver(
    beta,
    n_particles,
    grid,
    pot_g,
    n_samples=1000,
    step_size=0.1,
    seed=0,
    save_samples=False,
):
    if rank == 0:
        print(f"# Number of MPI ranks: {size}\n#")

    system_data = system(beta=beta, phi=pot_g, grid=grid, n_particles=n_particles)
    key = random.PRNGKey(seed + rank)
    key, random_key = random.split(key)
    random_numbers = random.uniform(random_key, shape=(n_samples,))
    key, random_key = random.split(key)
    random_proposals_r = step_size * random.normal(random_key, shape=(n_samples,))
    key, random_key = random.split(key)
    random_proposals_i = step_size * random.normal(random_key, shape=(n_samples,))
    random_proposals = random_proposals_r + 1.0j * random_proposals_i
    key, random_key = random.split(key)
    random_indices = random.randint(
        random_key, shape=(n_samples,), minval=0, maxval=len(grid.unique_indices)
    )
    random_data = (random_numbers, random_proposals, random_indices)

    # fields_data = initialize_fields(system_data)
    # rdf_sample = calc_rdf(fields_data['fields'], system_data)
    # exit()
    samples, acceptance_ratio = sampling(system_data, random_data)

    # mpi averaging
    global_energies = None
    global_phases = None
    # global_rdf = None
    if rank == 0:
        global_energies = np.zeros(n_samples, dtype=samples[1].dtype)
        global_phases = np.zeros(n_samples, dtype=samples[2].dtype)
        # global_rdf = np.zeros(n_bins, dtype=np.float64)
    comm.Reduce(
        [samples[1], MPI.COMPLEX], [global_energies, MPI.COMPLEX], op=MPI.SUM, root=0
    )
    comm.Reduce(
        [samples[2], MPI.COMPLEX], [global_phases, MPI.COMPLEX], op=MPI.SUM, root=0
    )
    # comm.Reduce([rdf_avg, MPI.DOUBLE], [
    #            global_rdf, MPI.DOUBLE], op=MPI.SUM, root=0)

    if rank == 0:
        global_energies /= size
        global_phases /= size
        # global_rdf /= size
        mean_energy, _ = stat_utils.blocking_analysis(
            global_phases, global_energies / n_particles, neql=100, printQ=True
        )
        print(f"Acceptance ratio: {acceptance_ratio}")
        # np.savetxt('rdf_avg.dat', global_rdf)

        # constant terms in potential energy
        nkpts_fields = len(grid.unique_indices) * 2
        avg_phase = jnp.average(global_phases)
        print(f"Average phase: {avg_phase}")
        pot_e = (
            mean_energy
            + (
                n_particles**2 * phi[grid.origin] / 2.0
                - n_particles / 2.0
                + nkpts_fields / 2 / beta
            )
            / n_particles
        )
        print(f"Potential energy: {pot_e}")

        # rdf
        r_points = jnp.linspace(1.0e-3, 1.0, 1000)
        rhog_2_avg = jnp.einsum("ij,i->j", samples[4], global_phases) / jnp.sum(
            global_phases
        )
        np.savetxt("rhog_2_avg.dat", rhog_2_avg.real)
        # 1d
        coskr = vmap(lambda g: jnp.cos(2.0 * jnp.pi * jnp.linalg.norm(g) * r_points))(
            jnp.array(grid.unique_k_points)
        )
        rdf = (
            2
            * n_particles
            * (n_particles - 1)
            * jnp.einsum("gr,g->r", coskr, rhog_2_avg).real
            + n_particles**2
        ) / n_particles**2
        # 3d
        # singr = vmap(
        #    lambda g: jnp.sin(2.0 * jnp.pi * jnp.linalg.norm(g) * r_points)
        #    / r_points
        #    / jnp.linalg.norm(g)
        #    / 2.0
        #    / jnp.pi
        # )(jnp.array(grid.unique_k_points))
        # rdf = (
        #    n_particles
        #    * (n_particles - 1)
        #    * jnp.einsum("gr,g->r", singr, rhog_2_avg).real
        #    + n_particles**2
        #    # - n_particles
        # ) / n_particles**2
        np.savetxt("rdf_avg.dat", np.stack((r_points * grid.l, rdf)).T)

    if save_samples:
        with open(f"samples_{rank}.pkl", "wb") as f:
            pickle.dump(samples, f)

    return samples, acceptance_ratio


if __name__ == "__main__":
    import grids

    l = 50.0
    beta = 4.0
    rho = 1.0
    n_particles = 50
    e_cut = 10000.0
    nkpts = 75
    grid = grids.one_dimensional_grid((nkpts,), e_cut=e_cut, l=l, r_grid_spacing=0.01)
    phi = jnp.array(
        [
            np.exp(-((2 * n * np.pi / l) ** 2) / 4) * np.pi**0.5 / l
            for n in range(-nkpts // 2 + 1, nkpts // 2 + 1)
        ]
    )
    n_samples = 400000
    samples, acceptance_ratio = driver(
        beta,
        n_particles,
        grid,
        phi,
        n_samples=n_samples,
        step_size=0.02,
        seed=101,
        save_samples=True,
    )
