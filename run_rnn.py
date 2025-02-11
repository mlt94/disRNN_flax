from disentangled_rnns.library import two_armed_bandits

from flax import nnx
import jax.numpy as jnp
import jax
from jax.example_libraries import optimizers 
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
    n_steps_per_session=500,
    n_sessions=50,
    batch_size=50,
)#returns [timestep, episode, feature]


def categorical_log_likelihood(labels, output_logits): 
    #mask = jnp.logical_not(labels < 0)
    log_probs = output_logits 
    #log_probs = jax.nn.log_softmax(output_logits)
    one_hot_labels = jax.nn.one_hot(
        labels[:, :, 0], num_classes=output_logits.shape[-1]
    )
    log_liks = one_hot_labels * log_probs
    #masked_log_liks = jnp.multiply(log_liks, mask)
    loss = jnp.mean(log_liks)
    return loss

def penalized_categorical_loss(model, x, y):
    _, model_output = model(x) #outupts carry, y, where y contains two features in its last dimension, being the predicted y scores and the penalty
    output_logits = model_output[:, :, :-1]
    penalty = jnp.sum(model_output[:, :, -1]) 
    loss = (
        categorical_log_likelihood(y, output_logits) * penalty
    )
    return loss


model = dis_rnn_model(rngs=nnx.Rngs(0))

optimizer = nnx.Optimizer(model, optax.adam(1e-3))


metrics = nnx.MultiMetric(
  loss=nnx.metrics.Average('loss'),
)


def train_step(model, optimizer:nnx.Optimizer, metrics:nnx.MultiMetric, x, y):
    grad_fn = nnx.value_and_grad(penalized_categorical_loss)
    loss, grads = grad_fn(model, x, y)
    metrics.update(loss=loss, labels=y)
    optimizer.update(grads)

metrics_history = {
  'train_loss': []
}

epochs = 15
x, y = next(dataset_train)
for epoch in range(epochs):
    model.train()
    train_step(model, optimizer, metrics, x, y)
  
    for metric, value in metrics.compute().items():  
        metrics_history[f'train_{metric}'].append(value.item())  
    metrics.reset() 
plt.plot(range(epochs), metrics_history["train_loss"])
plt.savefig("/home/mlut/disRNN_flax/train_loss.png")
