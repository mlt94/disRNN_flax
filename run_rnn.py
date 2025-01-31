from disentangled_rnns.library import two_armed_bandits

from flax import nnx
import jax.numpy as jnp
import jax
import optax
import numpy as np
import matplotlib.pyplot as plt

from source.model import own_rnn
from IPython import embed


'''Targets be a binary vector indicating the agent's choice on each trial, and inputs will consist of two binary vectors indicating the choice and reward from the previous trial.'''
agent = two_armed_bandits.AgentQ(alpha=0.3, beta=3)
environment = two_armed_bandits.EnvironmentBanditsDrift(sigma=0.1)
dataset_train = two_armed_bandits.create_dataset(
    agent,
    environment,
    n_steps_per_session=3,
    n_sessions=3,
    batch_size=3,
)

x, y = next(dataset_train) #return one batch of the data [timestep, episode, feature]

model = own_rnn(2, 5, rngs=nnx.Rngs(0)) 

def loss_fn(model: own_rnn, x, y):
    probs = model(x)
    loss = optax.sigmoid_binary_cross_entropy(probs.reshape(-1, 1), y.reshape(-1)).mean()
    return loss, probs

optimizer = nnx.Optimizer(model, optax.adam(0.01))
metrics = nnx.MultiMetric(
  accuracy=nnx.metrics.Accuracy(),
  loss=nnx.metrics.Average('loss'),
)


#@nnx.jit
def train_step(model: own_rnn, optimizer:nnx.Optimizer, metrics:nnx.MultiMetric, x, y):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, probs), grads = grad_fn(model, x, y)
    y = y.astype(jnp.int32)    
    metrics.update(loss=loss, logits=probs, labels=y.squeeze())
    optimizer.update(grads)

metrics_history = {
  'train_loss': [],
  'train_accuracy': [],
}

epochs = 1

for epoch in range(epochs):
    
    model.train()
    train_step(model, optimizer, metrics, x, y)

    for metric, value in metrics.compute().items():  
        metrics_history[f'train_{metric}'].append(value.item())  
    metrics.reset() 
plt.plot(range(epochs), metrics_history["train_loss"])
plt.ylim([0,1])
plt.savefig("/home/mlut/disRNN_flax/train_loss.png")
