"""The one episode loop. rl/run.py trains and evaluates with it; isaac/demo.py draws with it."""
import operator


def run_episode(env, policy, seed, train, on_step=None, layout=None):
    """Play one episode; returns env.totals plus mean_return.

    train: call policy.update after each step (and set policy.training).
    on_step(obs, actions, next_obs, info): called after each step, e.g. to draw it;
    returning True stops the episode early.
    layout: a new map string for this episode (None keeps the current map).
    """
    policy.training = train
    obs = env.reset(seed, layout)
    policy.spec = env.spec      # the map may have changed
    policy.reset()
    returns = [0.0] * env.n_robots
    done = False
    while not done:
        actions = _as_ints(policy.act(obs))
        next_obs, rewards, done, info = env.step(actions)
        if train:
            policy.update(obs, actions, rewards, next_obs, done)
        returns = [total + r for total, r in zip(returns, rewards)]
        stop = on_step is not None and on_step(obs, actions, next_obs, info)
        obs = next_obs
        if stop:
            break
    return dict(env.totals, mean_return=sum(returns) / len(returns))


def _as_ints(actions):
    """Accept numpy integers (np.argmax returns one); env.step rejects anything else that is not an int."""
    try:
        return [operator.index(a) for a in actions]
    except TypeError:
        return list(actions)
