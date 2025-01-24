from disentangled_rnns.library import two_armed_bandits

from flax import nnx
import jax.numpy as jnp
import optax


from source.model import own_rnn
from IPython import embed


'''Targets be a binary vector indicating the agent's choice on each trial, and inputs will consist of two binary vectors indicating the choice and reward from the previous trial.'''
agent = two_armed_bandits.AgentQ(alpha=0.3, beta=3)
environment = two_armed_bandits.EnvironmentBanditsDrift(sigma=0.1)
dataset_train = two_armed_bandits.create_dataset(
    agent,
    environment,
    n_steps_per_session=20,
    n_sessions=100,
    batch_size=100,
)

'''That which could be rewarding to investigate is the MLP + GRU part (no updates), plotting the five latents as in the figure from the paper'''
x, y = next(dataset_train) #return one batch of the data [timestep, episode, feature]
model = own_rnn(2, 4, rngs=nnx.Rngs(0))

def loss_fn(model: own_rnn, x, y):
    logits = model(x)
    loss = optax.sigmoid_binary_cross_entropy(logits.reshape(-1, 1), y.reshape(-1)).mean()
    return loss, logits

optimizer = nnx.Optimizer(model, optax.adam(0.01))
metrics = nnx.MultiMetric(
  accuracy=nnx.metrics.Accuracy(),
  loss=nnx.metrics.Average('loss'),
)

@nnx.jit
def train_step(model: own_rnn, optimizer:nnx.Optimizer, metrics:nnx.MultiMetric, x, y):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, x, y)
    y = y.astype(jnp.int32)    
    metrics.update(loss=loss, logits=logits, labels=y.squeeze())
    optimizer.update(grads)


@nnx.jit
def eval_step(model: own_rnn, metrics:nnx.MultiMetric, x, y):
    loss, logits = loss_fn(model, x, y)
    y = y.astype(jnp.int32)    
    metrics.update(loss=loss, logits=logits, labels=y.squeeze())


train_step(model, optimizer, metrics, x, y)

metrics_history = {
  'train_loss': [],
  'train_accuracy': [],
  'test_loss': [],
  'test_accuracy': [],
}

for metric, value in metrics.compute().items():  # Compute the metrics.
    metrics_history[f'train_{metric}'].append(value)  # Record the metrics.
metrics.reset()  # Reset the metrics for the test set.

eval_step(model, metrics, x, y)
for metric, value in metrics.compute().items():
    metrics_history[f'test_{metric}'].append(value)
metrics.reset()  # Reset the metrics for the next training epoch.
