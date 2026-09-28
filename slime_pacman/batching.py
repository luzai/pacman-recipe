"""Retain all episodes and pad only the DP schedule with zero-loss rows."""

from copy import copy, deepcopy


def pad_schedule(data, dp_size):
    """Align microbatch=1 to DP without dropping decisions or adding episodes."""
    count = len(data["tokens"])
    if not count or type(dp_size) is not int or dp_size <= 0:
        raise ValueError("invalid training batch or DP size")
    if any(
        not isinstance(values, list) or len(values) != count for values in data.values()
    ):
        raise ValueError("upstream conversion contract changed")
    for _ in range((-count) % dp_size):
        for values in data.values():
            values.append(deepcopy(values[0]))
        token = data["tokens"][0][-1]
        data["tokens"][-1] = [token, token]
        data["response_lengths"][-1] = 1
        data["loss_masks"][-1] = [0]
        data["rollout_log_probs"][-1] = [0.0]
        data["metadata"][-1].update(
            empty_episode=True, schedule_padding=True, allowed_token_ids=[]
        )
        if "multimodal_train_inputs" in data:
            data["multimodal_train_inputs"][-1] = None
    return data


def convert_samples(args, samples):
    from slime.rollout.batch_builder import BatchBuilder

    if args.micro_batch_size != 1 or args.use_dynamic_batch_size:
        raise ValueError("Pacman requires static microbatch=1")
    if (
        args.tensor_model_parallel_size,
        args.pipeline_model_parallel_size,
        args.context_parallel_size,
    ) != (1, 1, 1):
        raise ValueError("first Pacman adapter requires TP=PP=CP=1")
    if len({s.rollout_id for s in samples}) != args.global_batch_size:
        raise ValueError("expected one complete update of episodes")
    upstream_args = copy(args)
    upstream_args.custom_convert_samples_to_train_data_path = None
    data = BatchBuilder(upstream_args).convert(samples)
    return pad_schedule(data, args.actor_num_nodes * args.actor_num_gpus_per_node)
