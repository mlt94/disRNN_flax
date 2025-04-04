from disentangled_rnns.library import two_armed_bandits

from flax import nnx
import jax.numpy as jnp
import jax
from jax.example_libraries import optimizers as jax_optimizers 
import optax
import numpy as np
import matplotlib.pyplot as plt

from source.model import dis_rnn_model, dis_rnn_cell
from IPython import embed


'''Targets be a binary vector indicating the agent's choice on each trial, and inputs will consist of two binary vectors indicating the choice and reward from the previous trial.'''
agent = two_armed_bandits.AgentQ(alpha=0.3, beta=3)
environment = two_armed_bandits.EnvironmentBanditsDrift(sigma=0.1)
dataset_train = two_armed_bandits.create_dataset(
    agent,
    environment,
    n_steps_per_session=200,
    n_sessions=300,
    batch_size=300
)#returns [timestep, episode, feature]

def categorical_log_likelihood(labels, output_logits): 
    mask = jnp.logical_not(labels < 0)
    log_probs = jax.nn.log_softmax(output_logits)
    one_hot_labels = jax.nn.one_hot(
        labels[:, :, 0], num_classes=output_logits.shape[-1]
    )
    log_liks = one_hot_labels * log_probs
    masked_log_liks = jnp.multiply(log_liks, mask)
    loss = -jnp.nansum(masked_log_liks)
    return loss

def penalized_categorical_loss(model, x, y, penalty_scale=1e-3):
    _, model_output = model(x) #outupts carry, y, where y contains three features in its last dimension, being the predicted y scores for two classes and the penalty
    output_logits = model_output[:, :, :-1]
    penalty = jnp.sum(model_output[:, :, -1]) 
    loss = (
        categorical_log_likelihood(y, output_logits) + penalty_scale * penalty #penalty scale
    )
    return loss


model = dis_rnn_model(rngs=nnx.Rngs(0))

optimizer = nnx.Optimizer(model, optax.adam(1e-3)) #1e-3


metrics = nnx.MultiMetric(
  loss=nnx.metrics.Average('loss'),
)
@nnx.jit
def train_step(model, optimizer:nnx.Optimizer, metrics:nnx.MultiMetric, x, y):
    grad_fn = nnx.value_and_grad(penalized_categorical_loss)
    loss, grads = grad_fn(model, x, y)
    metrics.update(loss=loss, labels=y)
    clipped_grads = jax_optimizers.clip_grads(grads, 1e10)
    optimizer.update(clipped_grads) #calls optax.apply_updates internally

metrics_history = {
  'train_loss': []
}

x, y = next(dataset_train)
epochs = 100
for epoch in range(1, 1 + epochs):
    model.train()
    train_step(model, optimizer, metrics, x, y)
  
    for metric, value in metrics.compute().items():  
        normal_value = float(value.item()) 
        metrics_history[f'train_{metric}'].append(normal_value)
        print(f"{metric}: {normal_value:.2f}") 
    metrics.reset() 
    
plt.plot(range(1, 1 + epochs), metrics_history["train_loss"])
plt.savefig("/home/mlut/disRNN_flax/train_loss.png")
params = nnx.variables(model, nnx.Param)
print(params)

