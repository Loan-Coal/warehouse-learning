"""A checkpoint is a folder: the policy's own files plus meta.json written here.

meta.json records which policy, which env contract, which observation features and which
run flags (scenario, map, robots...) the weights belong to. So a checkpoint is never
silently loaded into the wrong algorithm or a changed env, and `--checkpoint FOLDER` alone
is enough to replay it with the settings it was trained on.
"""
import json
from pathlib import Path

from env.warehouse import CONTRACT_VERSION

META = "meta.json"


def save_checkpoint(policy, folder, name, n_robots, env_args=None, episode=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    policy.save(folder)
    meta = {"policy": name, "contract_version": CONTRACT_VERSION, "n_robots": n_robots,
            "obs": list(policy.spec.obs_features), "env_args": env_args or {}, "episode": episode}
    (folder / META).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def read_meta(folder):
    folder = Path(folder)
    if not (folder / META).is_file():
        raise ValueError(f"{folder} is not a checkpoint (no {META})")
    return json.loads((folder / META).read_text(encoding="utf-8"))


def load_checkpoint(policy, folder, name):
    folder = Path(folder)
    meta = read_meta(folder)
    if meta["policy"] != name:
        raise ValueError(f"checkpoint is for policy '{meta['policy']}', not '{name}'")
    if meta["contract_version"] != CONTRACT_VERSION:
        raise ValueError(f"checkpoint was trained on contract v{meta['contract_version']}, "
                         f"env is v{CONTRACT_VERSION}: retrain it")
    trained_obs = tuple(meta.get("obs", ()))
    if trained_obs != tuple(policy.spec.obs_features):
        raise ValueError(f"checkpoint was trained with OBS {trained_obs or 'unknown'}, the policy now has "
                         f"{tuple(policy.spec.obs_features)}: retrain it or restore its OBS")
    try:
        policy.load(folder)
    except FileNotFoundError as err:
        raise ValueError(f"checkpoint {folder} has no {Path(err.filename).name}: was it saved with the "
                         f"same registry entry (shared/independent) and version of '{name}'?") from err
    return meta
