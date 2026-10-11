"""Private CPU workers: no shared game map edits or CUDA contexts."""

import hashlib
import json
import os


class SeededSmallMazeClient:
    def __init__(self, client, context):
        self.client, self.context, self.index = client, context, 0

    async def post(self, *args, **kwargs):
        payload = dict(kwargs["json"])
        identity = [
            os.environ.get("SMALL_MAZE_REQUEST_SEED_DOMAIN", "map-training-v1"),
            self.context,
            self.index,
        ]
        raw = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        payload["sampling_params"] = dict(payload["sampling_params"])
        payload["sampling_params"]["sampling_seed"] = int.from_bytes(
            hashlib.sha256(raw).digest()[:4], "big"
        )
        kwargs["json"] = payload
        self.index += 1
        return await self.client.post(*args, **kwargs)


def worker_init(game_root, ghosts):
    os.environ.update(
        PACMAN_PYTHON_ROOT=game_root,
        SMALL_MAZE_GHOST_COUNT=str(ghosts),
        PACMAN_CUSTOM_WALL_RENDER="1",
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        SDL_VIDEODRIVER="dummy",
        SDL_AUDIODRIVER="dummy",
    )
    import torch

    torch.set_num_threads(1)


def collect_task(
    record, game_root, endpoint, hf_checkpoint, config_path, version, seed_context
):
    os.environ["PACMAN_PYTHON_ROOT"] = game_root
    from slime_pacman.rollout import collect_episode_in_worker, current_sources

    current_sources.cache_clear()
    sources = current_sources()
    if sources != record["source_revisions"]:
        raise ValueError("map worker source identity differs from record")
    return collect_episode_in_worker(
        record,
        endpoint,
        hf_checkpoint,
        config_path,
        sources,
        version,
        None,
        False,
        seed_context,
    )
