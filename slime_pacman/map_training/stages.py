"""Explicit curriculum-stage transition; model/optimizer payload stays untouched.

Call only after restoring and verifying the named parent joint checkpoint.
This resets curriculum statistics, never the global optimizer update counter.
"""
from copy import deepcopy


def begin_stage(parent, sampler, *, parent_identity, parent_update,
                receipt_sha256, expected_receipt_sha256, optimizer_budget):
    if (len(receipt_sha256) != 64 or receipt_sha256 != expected_receipt_sha256
            or parent.get('schema') != 'pacman-joint-checkpoint-v1'
            or parent.get('identity') != parent_identity
            or parent.get('update') != parent_update
            or parent['controller']['identity'] != parent_identity
            or parent['controller']['state']['last_update'] != parent_update):
        raise ValueError('curriculum stage parent identity/boundary mismatch')
    if not 0 <= parent_update < optimizer_budget - 1:
        raise ValueError('no remaining optimizer budget')
    controller = sampler.controller
    if (controller.state['last_update'] != -1
            or controller.identity == parent_identity):
        raise ValueError('stage requires a fresh distinct curriculum controller')
    state = deepcopy(controller.state)
    state['last_update'] = parent_update
    controller.state = state
    return dict(schema='pacman-curriculum-stage-v1', parent_identity=parent_identity,
                parent_receipt_sha256=receipt_sha256, parent_update=parent_update,
                completed_optimizer_updates=parent_update + 1,
                next_rollout=parent_update + 1, optimizer_budget=optimizer_budget,
                remaining_updates=optimizer_budget - parent_update - 1,
                new_identity=controller.identity, starting_tier=state['tier'],
                optimizer_reset=False, curriculum_statistics_reset=True)
