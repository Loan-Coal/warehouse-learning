"""Every algorithm the runners can use, by the name you pass to --policy.

To add yours: import it here and add one line (see "Writing an algorithm" in README.md).
Each value is called as value(spec, seed) and must return an rl.policy.Policy.
"""
from rl.policies._template import MyAlgo as TemplateQLearner
from rl.policies.central_dqn import CentralDQN
from rl.policies.greedy import GreedyRule
from rl.policy import independent, shared  # noqa: F401  (independent: for your entries)

POLICIES = {
    "greedy": shared(GreedyRule),
    "central_dqn": CentralDQN,              # one network for all robots, sees every robot and the map
    "template": shared(TemplateQLearner),   # the template as-is: try the whole pipeline before writing code
}
