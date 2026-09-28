"""A checkpoint is a folder: the policy's own files plus meta.json written here.

meta.json records which policy and which env contract the weights belong to, so a
checkpoint is never silently loaded into the wrong algorithm or a changed env.
"""
import json
from pathlib import Path

from env.warehouse import CONTRACT_VERSION

META = "meta.json"


def save_checkpoint(policy, folder, name, n_robots):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    policy.save(folder)
    meta = {"policy": name, "contract_version": CONTRACT_VERSION, "n_robots": n_robots}
    (folder / META).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_checkpoint(policy, folder, name):
    folder = Path(folder)
    if not (folder / META).is_file():
        raise ValueError(f"{folder} is not a checkpoint (no {META})")
    meta = json.loads((folder / META).read_text(encoding="utf-8"))
    if meta["policy"] != name:
        raise ValueError(f"checkpoint is for policy '{meta['policy']}', not '{name}'")
    if meta["contract_version"] != CONTRACT_VERSION:
        raise ValueError(f"checkpoint was trained on contract v{meta['contract_version']}, "
                         f"env is v{CONTRACT_VERSION}: retrain it")
    try:
        policy.load(folder)
    except FileNotFoundError as err:
        raise ValueError(f"checkpoint {folder} has no {Path(err.filename).name}: was it saved with the "
                         f"same registry entry (shared/independent) and version of '{name}'?") from err
    return meta
