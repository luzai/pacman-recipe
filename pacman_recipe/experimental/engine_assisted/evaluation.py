"""Explicit periodic checkpoint routing and immutable checkpoint identity checks."""
from pathlib import PurePosixPath
import re


def periodic_updates(*, inherited_updates, interval, final_updates):
    if any(type(value) is not int for value in (inherited_updates, interval, final_updates)) or inherited_updates < 0 or interval <= 0 or final_updates <= inherited_updates:
        raise ValueError('explicit increasing update budget and interval required')
    # The final checkpoint is evaluated separately, never duplicated here.
    return tuple(range(inherited_updates + interval, final_updates, interval))


def evaluation_key(checkpoint_sha256, domain, split):
    if not isinstance(checkpoint_sha256, str) or not re.fullmatch('[0-9a-f]{64}', checkpoint_sha256) or not domain or split not in ('validation', 'test'):
        raise ValueError('checkpoint hash, domain and sealed split required')
    return checkpoint_sha256, domain, split


def validate_checkpoint(checkpoint, *, root, protocol_sha256, maximum_updates):
    """Validate metadata only, without touching files or loading model weights."""
    base = PurePosixPath(root)
    path = PurePosixPath(checkpoint['checkpoint_path'])
    update = checkpoint['optimizer_updates']
    if not base.is_absolute() or '..' in base.parts or path.parent != base or '..' in path.parts or path.name != checkpoint['checkpoint_name']:
        raise ValueError('checkpoint must be directly inside explicit owned root')
    if type(maximum_updates) is not int or maximum_updates < 0 or type(update) is not int or not 0 <= update <= maximum_updates:
        raise ValueError('checkpoint outside explicit update budget')
    if checkpoint['protocol_sha256'] != protocol_sha256 or checkpoint['weight_version'] != 'language' + str(update):
        raise ValueError('checkpoint protocol/version mismatch')
    for key in ('checkpoint_sha256', 'adapter_sha256'):
        if not re.fullmatch('[0-9a-f]{64}', checkpoint[key]):
            raise ValueError('complete checkpoint/adapter identity required')
    return True
