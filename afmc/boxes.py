import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'True'
from jax import jit, numpy as jnp
from jax.tree_util import register_pytree_node_class
from typing import Sequence
from functools import partial
from dataclasses import dataclass
print = partial(print, flush=True)

@dataclass
@register_pytree_node_class
class one_dimensional_box():
    size: float = 1.
    length: float = None

    def __post_init__(self):
        self.length = (self.size,)

    @partial(jit, static_argnums=(0,))
    def get_ensemble_images(self, pos):
        return jnp.array([pos, pos + jnp.array(self.length)])

    @partial(jit, static_argnums=(0,))
    def get_particle_images(self, pos):
        return jnp.array([pos - jnp.array(self.length), pos, pos + jnp.array(self.length)])
    
    def get_init_pos(self, n_particles):
        return jnp.linspace(self.size / (n_particles+1), self.size, n_particles, endpoint=False).reshape(-1, 1)

    def __hash__(self):
        return hash((self.size, self.length,))

    def tree_flatten(self):
        return (), (self.size, self.length,)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*aux_data)


@dataclass
@register_pytree_node_class
class two_dimensional_box():
    size: float = 1.
    length: Sequence[float] = None
    
    def __post_init__(self):
        self.length = (self.size, self.size)

    @partial(jit, static_argnums=(0,))
    def get_ensemble_images(self, pos):
        return jnp.array(
            [pos + jnp.array(self.length) * jnp.array([0., 0.]),
             pos + jnp.array(self.length) * jnp.array([1., 0.]), 
             pos + jnp.array(self.length) * jnp.array([1., 1.]),
             pos + jnp.array(self.length) * jnp.array([0., 1.]), 
             pos + jnp.array(self.length) * jnp.array([-1., 1.])])

    @partial(jit, static_argnums=(0,))
    def get_particle_images(self, pos):
        return jnp.array(
            [pos + jnp.array(self.length) * jnp.array([0., 0.]),
             pos + jnp.array(self.length) * jnp.array([1., 0.]),
             pos + jnp.array(self.length) * jnp.array([1., 1.]),
             pos + jnp.array(self.length) * jnp.array([0., 1.]),
             pos + jnp.array(self.length) * jnp.array([-1., 1.]),
             pos + jnp.array(self.length) * jnp.array([-1., 0.]),
             pos + jnp.array(self.length) * jnp.array([-1., -1.]),
             pos + jnp.array(self.length) * jnp.array([0., -1.]),
             pos + jnp.array(self.length) * jnp.array([1., -1.])]) 

    def get_init_pos(self, n_particles):
        n_particles = int(n_particles**0.5)
        points = jnp.linspace(self.size / (n_particles+1),
                              self.size, n_particles, endpoint=False)
        pos = jnp.dstack(jnp.meshgrid(points, points)).reshape(-1, 2)
        return pos

    def __hash__(self):
        return hash((self.size, self.length,))

    def tree_flatten(self):
        return (), (self.size, self.length,)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*aux_data)
    

@dataclass
@register_pytree_node_class
class three_dimensional_box():
    size: float = 1.
    length: Sequence[float] = None
    
    def __post_init__(self):
        self.length = (self.size, self.size, self.size)

    @partial(jit, static_argnums=(0,))
    def get_ensemble_images(self, pos):
        return jnp.array(
            [pos + jnp.array(self.length) * jnp.array([0., 0.]),
             pos + jnp.array(self.length) * jnp.array([1., 0.]),
             pos + jnp.array(self.length) * jnp.array([1., 1.]),
             pos + jnp.array(self.length) * jnp.array([0., 1.]),
             pos + jnp.array(self.length) * jnp.array([-1., 1.])])

    @partial(jit, static_argnums=(0,))
    def get_particle_images(self, pos):
        return jnp.array(
            [pos + jnp.array(self.length) * jnp.array([0., 0., 0.]),
             pos + jnp.array(self.length) * jnp.array([1., 0., 0.]),
             pos + jnp.array(self.length) * jnp.array([1., 1., 0.]),
             pos + jnp.array(self.length) * jnp.array([0., 1., 0.]),
             pos + jnp.array(self.length) * jnp.array([-1., 1., 0.]),
             pos + jnp.array(self.length) * jnp.array([-1., 0., 0.]),
             pos + jnp.array(self.length) * jnp.array([-1., -1., 0.]),
             pos + jnp.array(self.length) * jnp.array([0., -1., 0.]),
             pos + jnp.array(self.length) * jnp.array([1., -1., 0.]),
             pos + jnp.array(self.length) * jnp.array([0., 0., 1.]),
             pos + jnp.array(self.length) * jnp.array([1., 0., 1.]),
             pos + jnp.array(self.length) * jnp.array([1., 1., 1.]),
             pos + jnp.array(self.length) * jnp.array([0., 1., 1.]),
             pos + jnp.array(self.length) * jnp.array([-1., 1., 1.]),
             pos + jnp.array(self.length) * jnp.array([-1., 0., 1.]),
             pos + jnp.array(self.length) * jnp.array([-1., -1., 1.]),
             pos + jnp.array(self.length) * jnp.array([0., -1., 1.]),
             pos + jnp.array(self.length) * jnp.array([1., -1., 1.]),
             pos + jnp.array(self.length) * jnp.array([0., 0., -1.]),
             pos + jnp.array(self.length) * jnp.array([1., 0., -1.]),
             pos + jnp.array(self.length) * jnp.array([1., 1., -1.]),
             pos + jnp.array(self.length) * jnp.array([0., 1., -1.]),
             pos + jnp.array(self.length) * jnp.array([-1., 1., -1.]),
             pos + jnp.array(self.length) * jnp.array([-1., 0., -1.]),
             pos + jnp.array(self.length) * jnp.array([-1., -1., -1.]),
             pos + jnp.array(self.length) * jnp.array([0., -1., -1.]),
             pos + jnp.array(self.length) * jnp.array([1., -1., -1.])])

    def get_init_pos(self, n_particles):
        n_particles = int(n_particles**(1./3.))
        points = jnp.linspace(self.size / (n_particles+1),
                              self.size, n_particles, endpoint=False)
        pos = jnp.dstack(jnp.meshgrid(points, points, points)).reshape(-1, 3)
        return pos
    
    def __hash__(self):
        return hash((self.size, self.length,))

    def tree_flatten(self):
        return (), (self.size, self.length,)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*aux_data)


if __name__ == '__main__':
    box = one_dimensional_box(1.)
    init_pos = box.get_init_pos(3)
    print(init_pos)