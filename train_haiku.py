from disentangled_rnns.library import two_armed_bandits
from disentangled_rnns.library.disrnn import HkDisRNN
from flax import nnx
import jax.numpy as jnp
import jax
import optax
import numpy as np
import haiku as hk
import matplotlib.pyplot as plt
from jax.example_libraries import optimizers
import chex

from IPython import embed


'''Targets be a binary vector indicating the agent's choice on each trial, and inputs will consist of two binary vectors indicating the choice and reward from the previous trial.'''
agent = two_armed_bandits.AgentQ(alpha=0.3, beta=3)
environment = two_armed_bandits.EnvironmentBanditsDrift(sigma=0.1)
dataset_train = two_armed_bandits.create_dataset(
    agent,
    environment,
    n_steps_per_session=200,
    n_sessions=300,
    batch_size=300,
)#returns [timestep, episode, feature]

update_mlp_shape = (5,5,5)
choice_mlp_shape = (2,2)
latent_size = 5

def make_network():
  return HkDisRNN(update_mlp_shape=update_mlp_shape,
                        choice_mlp_shape=choice_mlp_shape,
                        latent_size=latent_size,
                        obs_size=2, target_size=2)

def unroll_network(xs):
    core = make_network()
    batch_size = jnp.shape(xs)[1]
    state = core.initial_state(batch_size)
    ys, _ = hk.dynamic_unroll(core, xs, state)
    return ys

model = hk.transform(unroll_network)


def categorical_log_likelihood(
      labels: np.ndarray, output_logits: np.ndarray): #output_logits of dimensionality, n_steps, sequences, features
    # Mask any errors for which label is negative
    mask = jnp.logical_not(labels < 0)
    log_probs = jax.nn.log_softmax(output_logits)
    #lets inspect the shape of their output matrix
    if labels.shape[2] != 1:
      raise ValueError(
          'Categorical loss function requires targets to be of dimensionality'
          ' (n_timesteps, n_episodes, 1)'
      )
    one_hot_labels = jax.nn.one_hot(
        labels[:, :, 0], num_classes=output_logits.shape[-1]
    )
    log_liks = one_hot_labels * log_probs
    masked_log_liks = jnp.multiply(log_liks, mask)
    loss = -jnp.nansum(masked_log_liks)
    return loss

def penalized_categorical_loss(
      params, xs, targets, random_key, penalty_scale=0):
    """Treats the last element of the model outputs as a penalty."""
    # (n_steps, n_episodes, n_targets)
    model_output = model.apply(params, random_key, xs)
    output_logits = model_output[:, :, :-1]
    penalty = jnp.sum(model_output[:, :, -1])  # ()
    loss = (
        categorical_log_likelihood(targets, output_logits)
        + penalty_scale * penalty
    )
    return loss

def train_step(params, opt_state, xs, ys, random_key):
    loss, grads = jax.value_and_grad(penalized_categorical_loss, argnums=0)(
        params, xs, ys, random_key)
    
    grads, opt_state = opt.update(grads, opt_state)

    clipped_grads = optimizers.clip_grads(grads, 1e10)

    params = optax.apply_updates(params, clipped_grads)
    return loss, params, opt_state



epochs = 1000
random_key = jax.random.PRNGKey(0)
training_loss = []
random_key, key1 = jax.random.split(random_key)
xs_sample, _ = next(dataset_train)
params = model.init(key1, xs_sample)
opt = optax.adam(1e-3)
opt_state = opt.init(params)

xs, ys = next(dataset_train)
for epoch in range(epochs):
    random_key, key1, key2 = jax.random.split(random_key, 3)
    loss, params, opt_state = train_step(params, opt_state, xs, ys, key2)
    training_loss.append(float(loss))
plt.plot(range(epochs), training_loss)
plt.savefig("/home/mlut/disRNN_flax/train_loss_haiku.png")
