"""Publish a completion receipt only after a synchronous checkpoint is durable.

This verifies the bytes, not optimizer semantics. GPU recovery must additionally
demonstrate loaded optimizer/RNG state and the next completed update.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.pending-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        if os.name != 'nt':
            descriptor = os.open(path.parent, os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def publish(checkpoint_dir, controller_snapshot, *, identity, update, extra_files=()):
    root = Path(checkpoint_dir).resolve()
    marker = root / 'latest_checkpointed_iteration.txt'
    if int(marker.read_text().strip()) != update:
        raise ValueError('model checkpoint update mismatch')
    if controller_snapshot['state']['last_update'] != update:
        raise ValueError('controller update mismatch')
    model = root / f'iter_{update:07d}'
    if not model.is_dir():
        raise ValueError('checkpoint iteration directory missing')
    files = sorted(p for p in model.rglob('*') if p.is_file())
    if not files or any(p.stat().st_size == 0 for p in files):
        raise ValueError('empty checkpoint payload')
    files += [Path(p).resolve() for p in extra_files]
    receipts = []
    for file in files:
        relative = file.relative_to(root)
        with file.open('r+b') as stream:
            os.fsync(stream.fileno())
        receipts.append(dict(path=relative.as_posix(), size=file.stat().st_size, sha256=sha256(file)))
    with marker.open('r+b') as stream:
        os.fsync(stream.fileno())
    if os.name != 'nt':
        for parent in {p.parent for p in files} | {root}:
            descriptor = os.open(parent, os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    receipt = dict(schema='pacman-joint-checkpoint-v1', identity=identity, update=update,
                   controller=controller_snapshot, files=receipts)
    destination = root / 'joint' / f'completed-{update:07d}.json'
    if destination.exists() and json.loads(destination.read_text()) != receipt:
        raise ValueError('refuse replacement of a completed joint checkpoint')
    atomic_json(destination, receipt)
    # The receipt is durable before the latest joint pointer becomes visible.
    atomic_json(root / 'joint' / 'latest.json', dict(update=update, receipt_sha256=sha256(destination)))
    return receipt


def restore(checkpoint_dir, *, identity, update=None):
    root = Path(checkpoint_dir).resolve()
    pointer = json.loads((root / 'joint' / 'latest.json').read_text())
    chosen = pointer['update'] if update is None else update
    destination = root / 'joint' / f'completed-{chosen:07d}.json'
    if chosen == pointer['update'] and sha256(destination) != pointer['receipt_sha256']:
        raise ValueError('joint receipt changed')
    receipt = json.loads(destination.read_text())
    if (receipt['identity'] != identity or receipt['update'] != chosen
            or receipt['controller']['state']['last_update'] != chosen
            or int((root / 'latest_checkpointed_iteration.txt').read_text()) != chosen):
        raise ValueError('joint checkpoint identity/update mismatch')
    for row in receipt['files']:
        file = (root / row['path']).resolve()
        if not file.is_relative_to(root):
            raise ValueError('checkpoint file outside root')
        if file.stat().st_size != row['size'] or sha256(file) != row['sha256']:
            raise ValueError('checkpoint payload changed')
    return receipt
