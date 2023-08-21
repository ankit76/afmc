import os
import numpy as np
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'True'
from jax import grad, jit, vmap, random, lax, numpy as jnp
import jax
from flax import struct
from typing import Callable, Any
from functools import partial
print = partial(print, flush=True)
import pickle
import stat_utils
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


@struct.dataclass
class system:
    beta: float
    pot_fun: Callable = struct.field(pytree_node=False)
    box: Any = struct.field(pytree_node=False)
    n_particles: int = struct.field(pytree_node=False)
    n_bins: int = struct.field(pytree_node=False)
    r_max: float = struct.field(pytree_node=False)

@partial(jit, static_argnums=(0,))
def initialize_walker(system_data):
    box = system_data.box
    pot_fun = system_data.pot_fun
    pos = box.get_init_pos(system_data.n_particles)
    walker_data = {'pos': pos}    
    n_particles = pos.shape[0]
    images = box.get_ensemble_images(pos)
    displacements = vmap(
        lambda x: x[:, jnp.newaxis, :] - pos[jnp.newaxis, :, :], in_axes=0)(images)
    distances = vmap(lambda x: jnp.linalg.norm(x))(
        displacements.reshape(-1, pos.shape[-1])).reshape(images.shape[0], n_particles, n_particles)
    min_distances = vmap(lambda x: jnp.min(x))(distances.reshape(
        images.shape[0], -1).T).reshape(n_particles, n_particles)
    min_distances = jnp.where(
        min_distances < min_distances.T, min_distances, min_distances.T)
    energies = vmap(pot_fun)(min_distances)
    energy = jnp.sum(energies[jnp.triu_indices(n_particles, k=1)])
    walker_data['energies'] = energies
    walker_data['energy'] = energy
    walker_data['prop_energies'] = energies[0]
    walker_data['prop_energy_change'] = 0.
    walker_data['prop_pos'] = pos
    walker_data['n_accepted'] = 0
    walker_data['distances'] = min_distances
    walker_data['prop_distances'] = min_distances[0]
    walker_data['rdf_hist'] = jnp.histogram(
        min_distances[jnp.triu_indices(n_particles, k=1)], system_data.n_bins, (0., system_data.r_max))[0] / system_data.n_particles
    walker_data['prop_rdf_hist_change'] = jnp.zeros(system_data.n_bins)
    walker_data['rdf_hist_avg'] = jnp.zeros(system_data.n_bins)
    return walker_data

@partial(jit, static_argnums=(2,))
def prop_update_walker(walker_data, move, system_data):
    pos = walker_data['pos']
    box = system_data.box
    pot_fun = system_data.pot_fun
    n_particles = pos.shape[0]
    particle_id = move[0]
    new_pos_i = (pos[particle_id] + move[1]) % jnp.array(box.length)
    new_pos = pos.at[particle_id].set(new_pos_i)
    images = box.get_particle_images(new_pos[particle_id])
    displacements = vmap(lambda x: new_pos - x)(images)
    distances = vmap(lambda x: jnp.linalg.norm(x))(
        displacements.reshape(-1, pos.shape[-1])).reshape(images.shape[0], n_particles)
    min_distances = vmap(lambda x: jnp.min(x))(
        distances.reshape(images.shape[0], -1).T).reshape(n_particles)
    energies = vmap(pot_fun)(min_distances)
    old_energy = jnp.sum(walker_data['energies'][particle_id])
    energy_change = jnp.sum(energies) - old_energy
    walker_data['prop_energies'] = energies
    walker_data['prop_energy_change'] = energy_change
    walker_data['prop_pos'] = new_pos
    walker_data['prop_distances'] = min_distances
    rdf_hist_old = jnp.histogram(
        walker_data['distances'][move[0]], system_data.n_bins, (0., system_data.r_max))[0]
    rdf_hist_new = jnp.histogram(
        min_distances, system_data.n_bins, (0., system_data.r_max))[0]
    walker_data['prop_rdf_hist_change'] = (rdf_hist_new - rdf_hist_old) / system_data.n_particles
    return walker_data

@partial(jit, static_argnums=(3,))
def metropolis_step(walker_data, move, random_num, system_data):
    walker_data = prop_update_walker(walker_data, move, system_data)
    b_ratio = jnp.exp(-system_data.beta * walker_data['prop_energy_change'])
    new_pos = lax.cond(b_ratio > random_num, lambda x: walker_data['prop_pos'], lambda x: walker_data['pos'], 0)
    accepted = (b_ratio > random_num)
    new_energies = lax.cond(b_ratio > random_num, lambda x: x.at[move[0]].set(walker_data['prop_energies']), lambda x: x, walker_data['energies'])
    new_energies = lax.cond(b_ratio > random_num, lambda x: x.at[:,move[0]].set(walker_data['prop_energies']), lambda x: x, new_energies)
    new_distances = lax.cond(b_ratio > random_num, lambda x: x.at[move[0]].set(walker_data['prop_distances']), lambda x: x, walker_data['distances'])
    new_distances = lax.cond(b_ratio > random_num, lambda x: x.at[:,move[0]].set(walker_data['prop_distances']), lambda x: x, new_distances)
    walker_data['pos'] = new_pos
    walker_data['energies'] = new_energies
    walker_data['distances'] = new_distances
    walker_data['energy'] += accepted * walker_data['prop_energy_change']
    walker_data['rdf_hist'] += accepted * walker_data['prop_rdf_hist_change']
    walker_data['n_accepted'] += accepted
    return walker_data

@partial(jit, static_argnums=(0,))
def sampling(system_data, random_data):
    random_numbers, random_proposals, random_indices = random_data
    def scanned_fun(carry, x):
        random_num = random_numbers[x]
        random_proposal = random_proposals[x]
        random_index = random_indices[x]
        move = (random_index, random_proposal)
        carry = metropolis_step(carry, move, random_num, system_data)
        carry['rdf_hist_avg'] += (carry['rdf_hist'] - carry['rdf_hist_avg']) / (x + 1)
        return carry, (carry['pos'], carry['energy'])
    walker_data = initialize_walker(system_data)
    walker_data, samples = lax.scan(scanned_fun, walker_data, jnp.arange(random_numbers.shape[0]))
    return samples, walker_data['rdf_hist_avg'], walker_data['n_accepted'] / random_numbers.shape[0] 

def driver(beta, n_particles, box, pot_fun, n_samples=1000, r_max=None, n_bins=None, step_size=1., seed=0, save_samples=False):
    if rank == 0:
        print(f'# Number of MPI ranks: {size}\n#')

    if n_bins is None:
        n_bins = 50
    if r_max is None:
        r_max = 5 * box.length / n_particles
    system_data = system(beta=beta, pot_fun=pot_fun, box=box, n_particles=n_particles, n_bins=n_bins, r_max=r_max)
    key = random.PRNGKey(rank + seed)
    key, random_key = random.split(key)
    random_numbers = random.uniform(random_key, shape=(n_samples,))
    key, random_key = random.split(key)
    random_proposals = step_size * random.normal(random_key, shape=(n_samples,))
    key, random_key = random.split(key)
    random_indices = random.randint(random_key, shape=(n_samples,), minval=0, maxval=n_particles)
    random_data = (random_numbers, random_proposals, random_indices)
    samples, rdf_avg, accepted = sampling(system_data, random_data)
    
    # mpi averaging
    global_energies = None
    global_rdf = None
    if rank == 0:
        global_energies = np.zeros(n_samples, dtype=np.float64)
        global_rdf = np.zeros(n_bins, dtype=np.float64)
    comm.Reduce([samples[1], MPI.DOUBLE], [global_energies, MPI.DOUBLE], op=MPI.SUM, root=0)
    comm.Reduce([rdf_avg, MPI.DOUBLE], [global_rdf, MPI.DOUBLE], op=MPI.SUM, root=0)
    
    if rank == 0:
        global_energies /= size
        global_rdf /= size
        stat_utils.blocking_analysis(None, global_energies / n_particles, neql=100, printQ=True)
        print(f'Acceptance ratio: {accepted}')
        np.savetxt('rdf_avg.dat', global_rdf)
    
    if save_samples:
        with open(f'samples_{rank}.pkl', 'wb') as f:
            pickle.dump(samples, f)
        
    return samples, rdf_avg, accepted

if __name__ == '__main__':
    import boxes
    box = boxes.one_dimensional_box(50.)
    pot_fun = lambda x: jnp.exp(-jnp.abs(x)**2/2)
    #pot_fun = lambda x: jnp.abs(x)**2
    n_particles = 50
    n_samples = 100000
    beta = 50.
    samples, rdf_avg, accepted = driver(beta, n_particles, box, pot_fun, n_samples=n_samples, step_size=0.5)

    #walker_data = calculate_energy(walker_data, system_data)
    #print(walker_data)
    #exit()
    #move = (0, 0.1)
    #walker_data, accepted = metropolis_step(walker_data, (0, 0.1), 0.9, system_data)
    #print(walker_data)
    #print(accepted)
    exit()
    walker_data  = update_energy(walker_data, move, system_data)
    print(walker_data)
    exit()
    new_energy, energies = calculate_energy(pos.at[0].add(0.1), pot_fun, box)
    print(new_energy)