from flax import nnx
import jax
import jax.numpy as jnp

from flax.nnx.nn import initializers
from flax.nnx.nn.linear import Linear
from flax.nnx.nn.recurrent import RNN
default_kernel_init = initializers.lecun_normal()

from IPython import embed

class Block(nnx.Module):
  def __init__(self, input_dim, features, rngs):
    self.linear = nnx.Linear(input_dim, features, rngs=rngs)
    
  def __call__(self, x: jax.Array):  
    x = self.linear(x)
    x = jax.nn.relu(x)
    return x   
  
class MLP(nnx.Module):
  def __init__(self, features, num_layers, rngs):
    @nnx.split_rngs(splits=num_layers)
    @nnx.vmap(in_axes=(0,), out_axes=0)
    def create_block(rngs: nnx.Rngs):
      return Block(features, features, rngs=rngs)

    self.blocks = create_block(rngs)
    self.num_layers = num_layers

  def __call__(self, x):
    @nnx.split_rngs(splits=self.num_layers)
    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
    def forward(x, model):
      x = model(x)
      return x

    return forward(x, self.blocks)

def kl_gaussian(mean, var):
    return 0.5 * jnp.sum(-jnp.log(var) - 1.0 + var + jnp.square(mean), axis=-1)

class dis_rnn_cell(nnx.Module):
    def __init__(
        self,
        obs_size = 2, 
        target_size = 1, 
        latent_size = 5, # from 10
        update_mlp_shape = (5,5,5), #from 10,10,10
        choice_mlp_shae = (5,5,5), #from 10,10,10
        activation = jax.nn.relu,
        rngs = nnx.Rngs,
    ):
        self._target_size = target_size
        self._latent_size = latent_size
        self._update_mlp_shape = update_mlp_shape
        self._choice_mlp_shape = choice_mlp_shae
        self._activation = activation
        self._rngs = rngs

        mlp_input_size = latent_size + obs_size 

        initialize_update_mlp_sigmas = initializers.truncated_normal(lower=-3, upper=-2)(jax.random.key(42), (mlp_input_size, latent_size))
        update_mlp_sigmas_unsquashed = nnx.Param(initialize_update_mlp_sigmas) #equivalent to lines 75-79

        self._update_mlp_sigmas = nnx.Param(2 * jax.nn.sigmoid(update_mlp_sigmas_unsquashed)) #equivalent to line 81-83

        initialize_update_mlp_multipliers = initializers.constant(1)(jax.random.key(42), (mlp_input_size, latent_size))
        self._update_mlp_multipliers = nnx.Param(initialize_update_mlp_multipliers) #equivalent to line 84-88


        initialize_latent_sigmas_unsquashed = initializers.truncated_normal(lower=-3, upper=-3)(jax.random.key(42), latent_size,)
        self.latent_sigmas_unsquashed = nnx.Param(initialize_latent_sigmas_unsquashed) #equivalent to line 91-95
        
        self._latent_sigmas = nnx.Param(2 * nnx.sigmoid(self.latent_sigmas_unsquashed)) #equiavlent to line 96-98
        
        initialize_latents = initializers.truncated_normal(lower=-1, upper=1)(jax.random.key(42), (latent_size,))
        self._latent_inits = nnx.Param(initialize_latents)
        
    def __call__(self, observations, prev_latents): 
        penalty = 0 

        update_mlp_mus_unscaled = jnp.concatenate((observations, prev_latents), axis=1) #line 118-120
        #should have dimensions: observations shape (sequences,features) and prev_latents shape (sequences,hidden_size)
        
        update_mlp_mus = (jnp.expand_dims(update_mlp_mus_unscaled, 2) * self._update_mlp_multipliers) #line 122-125
        
        update_mlp_sigmas = self._update_mlp_sigmas #equivalent to line 127

        update_mlp_inputs = update_mlp_mus + update_mlp_sigmas * jax.random.normal(
            jax.random.key(42), update_mlp_mus.shape
        ) #equivalent to lines 129-131 (sequences, latent_size + features, latent_size)
        
        new_latents = jnp.zeros(shape=(prev_latents.shape)) #line 133

        for mlp_i in jnp.arange(self._latent_size): #equivalent to line 136-150
            penalty += kl_gaussian(update_mlp_mus[:,:, mlp_i], update_mlp_sigmas[:, mlp_i])
            
            update_mlp_output = MLP(self._update_mlp_shape, rngs=self._rngs)(update_mlp_inputs[:,:,mlp_i])



    def initialize_carry(self, batch_dims):
        mem_shape = (batch_dims, self._latent_size,)
        h = initializers.zeros(jax.random.key(42), mem_shape)
        return h
    
    @property
    def num_feature_axes(self) -> int:
        return 1

class dis_rnn_model(nnx.Module):
    def __init__(self, rngs=nnx.Rngs()):
        self.cell = dis_rnn_cell()
        self.linear = nnx.Linear(in_features = 5, out_features=2, rngs=rngs)
    def __call__(self, x):
        carry = self.cell.initialize_carry((x.shape[1]))
        scan_fn = lambda carry, cell, x: cell(x, carry)
        carry, y = nnx.scan(
           scan_fn, in_axes=(nnx.Carry, None, 0), out_axes=(nnx.Carry, 1)
        )(carry, self.cell, x) 
        x = self.linear(x)
        return jax.nn.sigmoid(x)








# class own_rnn(nnx.Module):
#     def __init__(self, din:int, dmid:int, rngs=nnx.Rngs):
#         self.cell = dis_rnn_cell()
#         self.linear = nnx.Linear(in_features = dmid, out_features=2,  rngs=rngs)
#     def __call__(self, x):
#         carry = self.cell.initialize_carry((x.shape[:-1]))
#         scan_fn = lambda carry, cell, x: cell(carry, x)
#         carry, y = nnx.scan(
#             scan_fn, in_axes=(nnx.Carry, None, 1), out_axes=(nnx.Carry, 1)
#         )(carry, self.cell, x) 
#         x = self.linear(y)
#         return jax.nn.sigmoid(x)


# class own_rnn(nnx.Module):
#     def __init__(self, input_size:int, hidden_size:int, rngs: nnx.Rngs):
#         self.cell = GRUCell(in_features=input_size, hidden_features=hidden_size, rngs=rngs)
#         self.rnn = RNN(self.cell, time_major=True) #the dataset has dimensions time_steps, sequences, features, so time is the major axis
#         self.classify = nnx.Linear(hidden_size, 1, rngs=rngs) #output classes

#     def __call__(self, x): 
#         x = self.rnn(x)
#         return jax.nn.sigmoid(self.classify(x))
    





# class RNNCell(nnx.Module):
#     def __init__(self, input_size, hidden_size, rngs):
#         self.linear = nnx.Linear(hidden_size + input_size, hidden_size, rngs=rngs)
#         self.hidden_size = hidden_size

#     def __call__(self, carry, x):
#         x = jnp.concatenate([carry, x], axis=-1)
#         x = self.linear(x)
#         x = jax.nn.relu(x)
#         return x, x
    
#     def initial_state(self, n_steps: int):
#         return jnp.zeros((n_steps, self.hidden_size))
    

# class RNN(nnx.Module):
#     def __init__(self, input_size:int, hidden_size:int, rngs: nnx.Rngs):
#         self.hidden_size = hidden_size
#         self.cell = RNNCell(input_size, self.hidden_size, rngs=rngs)
#         self.classify = nnx.Linear()

#     def __call__(self, x):
#         scan_fn = lambda carry, cell, x: cell(carry, x)
#         carry = self.cell.initial_state(x.shape[0]) #n_steps x features

#         carry, y = nnx.scan(
#             scan_fn, in_axes=(nnx.Carry, None, 1), out_axes=(nnx.Carry, 1)#in_axes and out_axes are positional arguments for nnx.scan
#         )(carry, self.cell, x) #nnx.scan is a transformation that takes as input a function
#         return y
    
