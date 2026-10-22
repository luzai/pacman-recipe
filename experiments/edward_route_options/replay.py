"""Inference-only paired native-runner suffix replay; no optimizer is constructed."""
import asyncio, base64, hashlib, io, json, os, sys
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from pacman_env.env.pygame_environment import PygamePacmanEnv, PygamePacmanEnvConfig
import pacman_recipe.level1.episode as native
from pacman_recipe.level1.trajectories import audit_trajectory
from pacman_recipe.level1 import prompts
import pacman_recipe.level1.trajectories as trajectory_module
E = Path(os.environ.get('EDWARD_MODEL_ROOT', '/prior'))
O = Path(os.environ.get('EDWARD_OUTPUT_ROOT', '/out'))
cpu = '--cpu' in sys.argv
rollout = os.environ.get('CPU_ROLLOUT') == '1'
os.environ['PACMAN_VISION_IMAGE_CONTRACT'] = 'cell-2x-bicubic-v1'
torch.set_num_threads(4)
p = AutoProcessor.from_pretrained(E / 'model160', local_files_only=True)
life_rule = prompts.IMAGE_PRIMITIVE_LIVES_SYSTEM.split('The state declares the life mode.', 1)[1].split('Choose an OPEN direction.', 1)[0].strip()
old_edward_system = prompts.edward_system_prompt

def edward_lives_system(*args, **kwargs):
    return old_edward_system(*args, **kwargs).replace('The episode ends on the first death.', 'The state declares the life mode. ' + life_rule)
prompts.edward_system_prompt = edward_lives_system
trajectory_module.edward_system_prompt = edward_lives_system
source = Path(native.__file__).read_text()
anchor = '\n            boundary_expected = None\n'
assert source.count(anchor) == 1
injection = '            replay_history = options.get("replay_history")\n            if replay_history is not None:\n                cell_exit_history = {tuple(x[:2]): list(x[2]) for x in replay_history["exits"]}\n                recent_actions = list(replay_history["actions"])\n                recent_positions = [tuple(x) for x in replay_history["positions"]]\n                no_progress_steps = replay_history["no_progress"]\n                if planner is not None:\n                    planner.remaining = {Position(*x) for x in replay_history["remaining"]}\n                    planner.last_action = recent_actions[-1] if recent_actions else None\n'
adapted = source.replace(anchor, '\n' + injection + anchor)
if '# EDWARD_ROUTE_HOOKS_V1' not in source:
    needle = '                if ascii_observation:\n                    from .prompts import render_ascii_map_for'
    assert adapted.count(needle) == 1
    adapted = adapted.replace(needle, '                if planner is not None and hasattr(planner, "route_menu"):\n                    state_context["route_menu"] = dict(planner.route_menu) if active_option is None else {}\n' + needle)
    needle = '                        active_objective_constraint = objective_constraint'
    assert adapted.count(needle) == 1
    adapted = adapted.replace(needle, '                        if hasattr(planner, "bind"):\n                            planner.bind(active_option, live_snapshot)\n' + needle)
    needle = '                next_image, base_reward, terminated, truncated, info = env.step(action)'
    assert adapted.count(needle) == 1
    adapted = adapted.replace(needle, '                if planner is not None and hasattr(planner, "before_action"):\n                    planner.before_action(action.value)\n' + needle)
exec(compile(adapted, str(O / 'native_episode_replay_adapter.py'), 'exec'), native.__dict__)
(O / 'native_episode_replay_adapter.py').write_text(adapted)
manifest = dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(), adapter_sha256=hashlib.sha256(adapted.encode()).hexdigest(), model_updates=160, source_rollouts=[160, 163], fallback='risk_ranked', sampling='greedy constrained argmax; probabilities at temperature0.7', training=False, next_training_clip_cov='0.002,1,5', scope='Five arms: native, native_step1, multi_step1, escape_step1, bound_segments. Same11 states, checkpoint160, RNG and budget. No training or LURE.')
(O / 'manifest.json').write_text(json.dumps(manifest, indent=2))
import pacman_env.planner as P
from .repair_v1 import RepairedPlanner as MultiStep1
from .repair_adapter import RepairedPlanner as EscapeStep1
from .route_bound import BoundPlanner, NativeStep1, install_prompt, audit_bound
install_prompt(prompts, native)
ARMS = {'native': P.EdwardPlanner, 'native_step1': NativeStep1, 'multi_step1': MultiStep1, 'escape_step1': EscapeStep1, 'bound_segments': BoundPlanner}
BANK = Path(os.environ['EDWARD_BANK_ROOT'])
cases = json.loads((BANK / 'cases.json').read_text())
assert len(cases) >= 8
m = None
if not cpu:
    assert json.loads((O / 'CPU_PASS.json').read_text())['passed']
    m = Qwen3_5ForConditionalGeneration.from_pretrained(E / 'model160', torch_dtype=torch.bfloat16, device_map='cuda', attn_implementation='sdpa', local_files_only=True).eval()

class FirstBoundary(Exception):
    pass

def env_factory(config):
    env = PygamePacmanEnv(config)
    original = env._saved_state_identity
    cache = []

    def identity():
        if not cache:
            cache.append(original())
        return cache[0]
    env._saved_state_identity = identity
    return env

def validate_replay_row(row):
    assert row['env']['max_steps'] == 512 and row['env']['ghost_mode'] == 'normal'
    assert row['env']['backend'] == 'original-pygame'
    assert len(row['restart_state_sha256']) == 64

class Runner(native.PacmanEpisodeRunner):

    async def _call_model(self, messages, **options):
        converted = []
        images = []
        for msg in messages:
            content = msg['content']
            if isinstance(content, list):
                cc = []
                for part in content:
                    if part['type'] == 'image_url':
                        raw = base64.b64decode(part['image_url']['url'].split(',', 1)[1])
                        images.append(Image.open(io.BytesIO(raw)).convert('RGB'))
                        cc.append(dict(type='image'))
                    else:
                        cc.append(part)
                content = cc
            converted.append(dict(role=msg['role'], content=content))
        prompt = p.apply_chat_template(converted, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        batch = p(text=[prompt], images=images, return_tensors='pt')
        constraint = options.get('objective_constraint')
        support = list(constraint.allowed_token_ids) if constraint is not None else [p.tokenizer.encode(x, add_special_tokens=False)[0] for x in options['current_open_actions']]
        if not self.records:
            (self.folder / 'first-prompt.txt').write_text(prompt)
            assert hashlib.sha256(raw).hexdigest() == self.case['image_sha256'], 'initial image mismatch'
            if self.mode == 'primitive':
                if batch['input_ids'][0].tolist() != self.case['input_ids']:
                    import difflib
                    print('PROMPT_DIFF', ''.join(difflib.unified_diff(self.case['prompt'].splitlines(True), prompt.splitlines(True))), flush=True)
                    print('TOKEN_LENGTHS', len(self.case['input_ids']), len(batch['input_ids'][0]), flush=True)
                    raise AssertionError('initial primitive prompt mismatch')
                assert set(support) == set(self.case['allowed_token_ids']), 'initial primitive support mismatch'
            (self.folder / 'first-prompt.txt').write_text(prompt)
            (self.folder / 'first-image.png').write_bytes(raw)
        if cpu and (not rollout):
            self.records.append(dict(support=support, input_tokens=batch['input_ids'].shape[1], image_match=True, primitive_prompt_match=self.mode == 'primitive'))
            raise FirstBoundary()
        if cpu and rollout:
            cs = self.boundary['context']['verify']['candidates']
            selected = min(cs, key=lambda x: ({'ELIMINATE': 0, 'COLLECT': 1, 'AVOID': 2}.get(x['strategy'], 3), int(x['option_id'][1:])))
            code = constraint.code_for_option(selected['option_id'])
            self.records.append(dict(code=code))
            return native.ModelTurn(code, f'cpu-{len(self.records)}', messages, raw_response={})
        batch = {k: v.to('cuda') for k, v in batch.items()}
        batch['pixel_values'] = batch['pixel_values'].to(torch.bfloat16)
        with torch.inference_mode():
            logits = m(**batch, use_cache=False, logits_to_keep=1).logits[0, -1].float()
        probs = torch.softmax(logits[support] / 0.7, dim=-1).cpu().tolist()
        idx = max(range(len(probs)), key=probs.__getitem__)
        code = p.tokenizer.decode([support[idx]], skip_special_tokens=False)
        record = dict(index=len(self.records), code=code, support=support, probabilities=probs, prompt=prompt)
        self.records.append(record)
        (self.folder / 'decisions.json').write_text(json.dumps(self.records))
        if len(self.records) % 20 == 0:
            print('PROGRESS', self.case['id'], self.mode, len(self.records), flush=True)
        return native.ModelTurn(code, f'replay-{len(self.records)}', messages, raw_response={'local_hf': record})

async def main():
    results = []
    for case in cases:
        episode = json.loads((E / 'episodes' / case['file']).read_text())
        steps = episode['trajectory']['episode']['trajectory']
        i = case['step'] - 1
        exits = {}
        actions = []
        positions = []
        dry = 0
        for step in steps[:i]:
            pos = tuple(step['observation_context']['pacman_position'])
            exits.setdefault(pos, []).append(step['action'])
            actions.append(step['action'])
            positions.append(list(pos))
            dry = 0 if step.get('normal_pellet_eaten') or step.get('event_game_score_delta', 0) > 0 else dry + 1
        positions.append(case['context']['pacman_position'])
        history = dict(exits=[[r, c, a] for (r, c), a in exits.items()], actions=actions, positions=positions, no_progress=dry, remaining=case['snapshot']['normal_pellet_positions'] + case['snapshot'].get('power_pellet_positions', []))
        os.environ.update(PACMAN_PYTHON_ROOT=case['game_root'], SMALL_MAZE_GHOST_COUNT='2', PACMAN_DISABLE_FRUIT='1')
        with PygamePacmanEnv(PygamePacmanEnvConfig(pacman_python_root=case['game_root'], max_steps=512, ghost_mode='normal', episode_life_mode='original_three_lives')) as probe:
            spec = probe.spec
            prov = probe.provenance
            requested = dict(name=spec.env_id, api_version=spec.api_version, backend='original-pygame', ghost_mode='normal', pacman_python_revision=prov['pacman_python_commit'], pacman_python_source_sha256=prov['pacman_python_source_sha256'], maapacman_revision=prov['maapacman_commit'], maapacman_env_source_sha256=prov['maapacman_env_source_sha256'], level_revision=spec.level_revision, ruleset_revision=spec.ruleset_revision, level=1, seed=case['seed'], max_steps=512)
        original_episode = episode['trajectory']['episode']
        row = dict(id=case['id'], split='test', env=requested, dataset_contract_version=original_episode['dataset_contract_version'], source_revisions=original_episode['source_revisions'], training_backend=original_episode['training_backend'])
        statepath = BANK / 'cases' / case['id'] / 'state.json'
        saved = json.loads(statepath.read_text())
        assert saved['sha256'] == case['state_sha256']
        row.update(restart_state_path=str(statepath), restart_state_sha256=hashlib.sha256(statepath.read_bytes()).hexdigest(), restart_state_id=case['id'])
        for mode, planner_class in ARMS.items():
            native.EdwardPlanner = planner_class
            folder = O / os.environ.get('REPLAY_PHASE', 'cpu' if cpu else 'gpu') / case['id'] / mode
            folder.mkdir(parents=True, exist_ok=False)
            runner = Runner(tokenizer=p.tokenizer, env_factory=env_factory, row_validator=validate_replay_row, edward_options=True, edward_harness='legacy', open_action_mask=False, edward_fallback_mode='risk_ranked', image_prompt_style='live_state_v3', episode_life_mode='original_three_lives', enable_thinking=False, pacman_python_root=case['game_root'], worker_base_dir='/tmp/edward-replay', replay_history=history, use_base_reward=False, wall_penalty=0.0, step_penalty=0.0, completion_reward=1.0)
            runner.records = []
            runner.case = case
            runner.mode = mode
            runner.folder = folder
            if True:

                def sink(boundary):
                    runner.boundary = boundary
                    if not (folder / 'first-boundary.json').exists():
                        (folder / 'first-boundary.json').write_text(json.dumps(boundary))
                runner.decision_boundary_sink = sink
            try:
                await runner.run(row)
            except FirstBoundary:
                assert cpu
            except trajectory_module.TrajectoryAuditError as exc:
                (folder / 'REJECTED_EPISODE.json').write_text(json.dumps(exc.payload))
                raise
            if cpu and (not rollout):
                assert len(runner.records) == 1
                result = dict(case=case['id'], mode=mode, passed=True, **runner.records[0])
            else:
                ep = runner.last_episode
                assert ep is not None
                (folder / 'episode.json').write_text(json.dumps(ep))
                audit_trajectory(ep)
                traj = ep['trajectory']
                start = saved['payload']['episode']['steps']
                assert len(traj) <= 512 - start
                result = dict(case=case['id'], mode=mode, remaining_start=case['snapshot']['normal_pellets'], start_env_step=start, steps=len(traj), decisions=len(runner.records), won=ep['won'], remaining_end=ep['normal_pellets_remaining'], new_deaths=ep['death_count'] - saved['payload']['episode']['death_count'], reason=ep['terminal_reason'], actions=''.join((x['action'] or '' for x in traj)))
            if not cpu or rollout:
                model_steps = [t for t in traj if t['model_called']]
                result['model_calls'] = len(model_steps)
                result['immediate_reversals'] = sum(({'U': 'D', 'D': 'U', 'L': 'R', 'R': 'L'}.get(a['action']) == b['action'] for a, b in zip(traj, traj[1:])))
                positions = [t['observation_context']['pacman_position'] for t in traj]
                result['aba_returns'] = sum((a == c for a, c in zip(positions, positions[2:])))
                result['objective_switches'] = sum(((a.get('option_strategy'), a.get('option_target')) != (b.get('option_strategy'), b.get('option_target')) for a, b in zip(model_steps, model_steps[1:])))
                result['route_switches'] = None
                if mode == 'bound_segments':
                    planner = BoundPlanner.last_instance
                    result['route_audit'] = audit_bound(ep, planner)
                    result['route_switches'] = sum((e.get('route_switch', False) for e in planner.events))
                    (folder / 'routes.json').write_text(json.dumps(dict(registry=planner.registry, events=planner.events)))
            results.append(result)
            print('RESULT', json.dumps(result), flush=True)
            (O / ('CPU_ROLLOUT_PARTIAL.json' if rollout else 'CPU_PARTIAL.json' if cpu else 'RESULTS_PARTIAL.json')).write_text(json.dumps(results, indent=2))
    (O / ('CPU_ROLLOUT.json' if rollout else 'CPU_PASS.json' if cpu else 'RESULTS.json')).write_text(json.dumps(dict(passed=True, results=results, manifest=manifest), indent=2))
asyncio.run(main())
