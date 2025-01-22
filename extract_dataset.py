from disentangled_rnns.library import get_datasets
from disentangled_rnns.library import two_armed_bandits

from IPython import embed

agent = two_armed_bandits.AgentQ(alpha=0.3, beta=3)
environment = two_armed_bandits.EnvironmentBanditsDrift(sigma=0.1)
dataset = two_armed_bandits.create_dataset(
    agent,
    environment,
    n_steps_per_session=20,
    n_sessions=10,
    batch_size=1,
)
embed()
