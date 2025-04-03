from flax import nnx
import jax
import jax.numpy as jnp

from flax.nnx.nn.linear import Linear
from flax.nnx.nn.recurrent import RNN
from flax.nnx.nn import initializers
default_kernel_init = initializers.lecun_normal()

from source.mlp import MLP, haiku_adapated_linear

from IPython import embed

def kl_gaussian(mean, var):
    return 0.5 * jnp.sum(-jnp.log(var) - 1.0 + var + jnp.square(mean), axis=-1)

class dis_rnn_cell(nnx.Module):
    def __init__(
        self,
        obs_size = 2, 
        target_size = 2, #from default 1
        latent_size = 5, # from 10
        update_mlp_shape = (5,5,5), #from 10,10,10
        choice_mlp_shae = (5,5,5), #from 10,10,10
        activation = jax.nn.relu,
        rngs =  nnx.Rngs,
    ):
        self._target_size = target_size
        self._latent_size = latent_size
        self._update_mlp_shape = update_mlp_shape
        self._choice_mlp_shape = choice_mlp_shae
        self._activation = activation
        self._rngs = rngs

        mlp_input_size = latent_size + obs_size 
        key = rngs.params()

        initialize_update_mlp_sigmas = initializers.truncated_normal(stddev = 1, lower=-3, upper=-2, dtype=jnp.float32)(key, (mlp_input_size, latent_size))
        update_mlp_sigmas_unsquashed = nnx.Param(initialize_update_mlp_sigmas, dtype=jnp.float32) #equivalent to lines 75-79
        
        self._update_mlp_sigmas = nnx.Variable(2 * nnx.sigmoid(update_mlp_sigmas_unsquashed.value)) #equivalent to line 81-83 

        initialize_update_mlp_multipliers = initializers.constant(1, dtype=jnp.float32)(key, (mlp_input_size, latent_size))
        self.update_mlp_multipliers = nnx.Param(initialize_update_mlp_multipliers, dtype=jnp.float32) #equivalent to line 84-88
        
        initialize_latent_sigmas_unsquashed = initializers.truncated_normal(stddev = 1, lower=-3, upper=-2, dtype=jnp.float32)(key, (latent_size, ))
        self.latent_sigmas_unsquashed = nnx.Param(initialize_latent_sigmas_unsquashed) #equivalent to line 91-95
        
        self._latent_sigmas = nnx.Variable(2 * nnx.sigmoid(self.latent_sigmas_unsquashed.value)) #equiavlent to line 96-98 
                
        initialize_latents = initializers.truncated_normal(stddev = 1, lower=-1, upper=1, dtype=jnp.float32)(key, (latent_size,))
        self.latent_inits = nnx.Param(initialize_latents)

        self.mlp = MLP(self._update_mlp_shape, rngs=self._rngs)
        self.haiku_adapated_linear = haiku_adapated_linear(1, rngs=self._rngs)
        
    def __call__(self, observations, prev_latents): 
        penalty = 0 

        update_mlp_mus_unscaled = jnp.concatenate((observations, prev_latents), axis=1) #line 118-120
        #should have dimensions: observations shape (sequences,features) and prev_latents shape (sequences,hidden_size)

        update_mlp_mus = (jnp.expand_dims(update_mlp_mus_unscaled, 2) * self.update_mlp_multipliers.value) #line 122-125
        
        update_mlp_sigmas = self._update_mlp_sigmas.value #equivalent to line 127

        update_mlp_inputs = update_mlp_mus + update_mlp_sigmas * jax.random.normal(
            self._rngs.params(), update_mlp_mus.shape
        ) #equivalent to lines 129-131 (sequences, latent_size + features, latent_size)

        new_latents = jnp.zeros(shape=(prev_latents.shape)) #line 133

        for mlp_i in jnp.arange(self._latent_size): #equivalent to line 136-150
            penalty += 1 * kl_gaussian(update_mlp_mus[:,:, mlp_i], update_mlp_sigmas[:, mlp_i])
            
            update_mlp_output = self.mlp(update_mlp_inputs[:,:,mlp_i]) #outputs (sequences, latent_size)
            
            update = self.haiku_adapated_linear(1, rngs=self._rngs)(update_mlp_output)[:,0]

            w = jax.nn.sigmoid(self.haiku_adapated_linear(1, rngs=self._rngs)(update_mlp_output))[:,0]

            new_latent = w * update + (1 - w) * prev_latents[:, mlp_i] #GRU Cell without reset gate
            new_latents = new_latents.at[:,mlp_i].set(new_latent) # inplace update in Jax

        noised_up_latents = new_latents + self._latent_sigmas.value * jax.random.normal(
            self._rngs.params(), new_latents.shape
        ) #lines 152-158
            
        penalty += kl_gaussian(new_latents, self._latent_sigmas.value) #line 159

        choice_mlp_output = MLP(self._choice_mlp_shape, rngs=self._rngs)(noised_up_latents) #166-168
        y_hat = haiku_adapated_linear(self._target_size, rngs=self._rngs)(choice_mlp_output) #170

        penalty = jnp.expand_dims(penalty, 1)
        output = jnp.concatenate((y_hat, penalty), axis=1)
        return noised_up_latents, output

    def initialize_carry(self, batch_dims):
        mem_shape = (batch_dims, self._latent_size,)
        h = initializers.ones_init()(self._rngs.params(), mem_shape) * self.latent_inits.value
        return h
    
    @property
    def num_feature_axes(self) -> int:
        return 1


class dis_rnn_model(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.cell = dis_rnn_cell(rngs=rngs)
    def __call__(self, x):
        carry = self.cell.initialize_carry((x.shape[1]))
        # y_s = []
        # for t in range(x.shape[0]):
        #     carry, y = self.cell(x[t,:, :], carry)
        #     y_s.append(y)
        # y = jnp.stack(y_s, axis=0)

        scan_fn = lambda carry, cell, x: cell(x, carry)
        carry, y = nnx.scan(
            scan_fn, in_axes=(nnx.Carry, None, 0), out_axes=(nnx.Carry, 0)
        )(carry, self.cell, x) 
        return carry, y
