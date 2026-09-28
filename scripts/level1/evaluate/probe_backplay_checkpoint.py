"""Probe saved restart states with a frozen HF checkpoint; perform no updates."""
from pathlib import Path
import argparse
import copy
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def run_probe(experiment, *, source_policy_version: str, candidate_ids: list[str]):
    """Reuse the training probe contract without installing training hooks."""
    requested = set(candidate_ids)
    available = {e['restart_state_id']: e for e in experiment.candidates}
    if len(requested) != len(candidate_ids):
        raise ValueError('duplicate candidate IDs')
    if not requested or requested - available.keys():
        raise ValueError('explicit known candidate IDs required')
    entries = [available[state_id] for state_id in candidate_ids]
    experiment.report.update(phase='checkpoint-probe', source_policy_version=source_policy_version,
                             optimizer_updates=0, pilot_updates=0, adaptive_updates=0,
                             candidate_ids=[e['restart_state_id'] for e in entries])
    experiment.persist()
    try:
        result = experiment.probe(entries, 'checkpoint-frontier')
        experiment.report.update(status='checkpoint_probe_completed', summaries=result)
        experiment.persist()
        return result
    except Exception:
        experiment.report['status'] = 'failed'
        experiment.persist()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--actor-checkpoint', type=Path, required=True)
    parser.add_argument('--source-policy-version', required=True)
    parser.add_argument('--candidate-id', action='append', required=True)
    parser.add_argument('--probe-samples', type=int, default=24)
    args, config_args = parser.parse_known_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    checkpoint = args.actor_checkpoint.resolve()
    if not (checkpoint / 'config.json').is_file():
        raise ValueError('actor checkpoint must be a complete HF model directory')
    if args.probe_samples < 24 or args.probe_samples % 12:
        raise ValueError('probe-samples must be >=24 and divisible by12')
    from train_areal import (_build_workflow_kwargs, _validate_reward_objective_contract,
                            _validate_backplay_contract, _validate_actor_colocated_reference_offload)
    from areal import PPOTrainer
    from areal.api.cli_args import load_expr_config
    from areal.utils.environ import is_single_controller
    from datasets import Dataset
    from omegaconf import OmegaConf
    from pacman_recipe.synthetic.configs import PacmanAgentConfig
    from pacman_recipe.level1.level1_dataset import make_episode_row
    from pacman_recipe.level1.backplay_runner import BackplayExperiment, load_restart_bank, redact_config, write_json
    # Apply before interpolation resolves: actor, tokenizer and vLLM must load
    # the same frozen model, rather than a stale training base model.
    config, _ = load_expr_config([*config_args, f'actor.path={checkpoint}',
                                  f'tokenizer_path={checkpoint}', f'vllm.model={checkpoint}',
                                  'recover.mode=disabled'], PacmanAgentConfig)
    if not is_single_controller() or config.actor.use_lora:
        raise ValueError('requires single-controller mode and a full/merged HF checkpoint')
    _validate_reward_objective_contract(config)
    _validate_backplay_contract(config)
    _validate_actor_colocated_reference_offload(config)
    bank = load_restart_bank(args.bank)
    known = {e['restart_state_id'] for e in bank['restart_states']}
    if set(args.candidate_id) - known:
        raise ValueError('unknown candidate ID')
    template = make_episode_row(1, split='validation', seed=int(bank['restart_states'][0]['seed']),
                               max_steps=config.environment.max_steps,
                               ghost_mode=config.environment.ghost_mode, action_protocol=config.action_protocol)
    rows = [dict(copy.deepcopy(template), id=f'probe-placeholder-{i}')
            for i in range(config.train_dataset.batch_size)]
    dataset = Dataset.from_list(rows)
    kwargs = _build_workflow_kwargs(config, config.eval_gconfig, training=False)
    # Initialization uses the same allocation as training. No train(), install(),
    # prepare_batch(), backward(), or optimizer step is invoked.
    with PPOTrainer(config, train_dataset=dataset, valid_dataset=dataset) as trainer:
        if trainer.recover_info is not None or trainer.eval_rollout is None:
            raise ValueError('requires fresh checkpoint initialization and eval rollout')
        experiment = BackplayExperiment(trainer=trainer, template=template, bank_dir=args.bank,
            output_dir=args.output, workflow=config.workflow, probe_kwargs=kwargs,
            candidate_ids=args.candidate_id, probe_samples=args.probe_samples)
        write_json(experiment.output_dir / 'resolved-config.json', redact_config(
            OmegaConf.to_container(OmegaConf.structured(config), resolve=True)))
        write_json(experiment.output_dir / 'input-template.json', template)
        if trainer._requires_proxy_workflow(config.workflow):
            trainer._ensure_proxy_started()
        run_probe(experiment, source_policy_version=args.source_policy_version,
                  candidate_ids=args.candidate_id)


if __name__ == '__main__':
    main()
