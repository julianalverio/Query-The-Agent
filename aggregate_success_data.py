import os
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def aggregate_success_data(path):
    steps_path = os.path.join(path, 'steps.npy')
    values_path = os.path.join(path, 'success_rate.npy')
    stds_path = os.path.join(path, 'stds.npy')
    num_seeds_path = os.path.join(path, 'num_seeds.npy')
    if os.path.exists(steps_path) and os.path.exists(values_path) and os.path.exists(stds_path) and os.path.exists(num_seeds_path):
        return np.load(steps_path), np.load(values_path), np.load(stds_path), np.load(num_seeds_path)

    target_tag = 'test/Eval Success Rate'
    new_target_tag = 'eval/Eval Success Rate'

    subdirs = [subdir for subdir in os.listdir(path) if os.path.isdir(os.path.join(path, subdir))]

    # read in data
    summary_iterators = list()
    for sub_directory in subdirs:
        subdir_path = os.path.join(path, sub_directory)
        if not os.path.isdir(subdir_path):
            continue
        accumulator = EventAccumulator(subdir_path).Reload()
        if target_tag in accumulator.Tags()['scalars']:
            summary_iterators.append(accumulator)
        elif new_target_tag in accumulator.Tags()['scalars']:
            summary_iterators.append(accumulator)

    # find common steps
    common_steps = None
    for iterator in summary_iterators:
        if target_tag in iterator.Tags()['scalars']:
            local_target_tag = target_tag
        else:
            local_target_tag = new_target_tag
        steps = set([e.step for e in iterator.Scalars(local_target_tag)])
        if common_steps is None:
            common_steps = steps
        common_steps = common_steps.intersection(steps)
        intersection = common_steps.intersection(steps)

    all_values = list()
    for idx, iterator in enumerate(summary_iterators):
        values = list()
        if target_tag in iterator.Tags()['scalars']:
            local_target_tag = target_tag
        else:
            local_target_tag = new_target_tag
        for event in iterator.Scalars(local_target_tag):
            if event.step in common_steps:
                values.append(event.value)
        if len(values) != len(common_steps):
            msg = f'Problem! Expected length {len(common_steps)} but got length {len(values)} at {subdirs[idx]}'
            raise ValueError(msg)
        all_values.append(values)

    all_values = np.array(all_values)
    mean_values = all_values.mean(axis=0)
    std_values = all_values.std(axis=0)

    num_seeds = np.array(len(summary_iterators))

    common_steps = np.array(sorted(list(common_steps)))
    np.save(steps_path, common_steps)
    np.save(values_path, mean_values)
    np.save(stds_path, std_values)
    np.save(num_seeds_path, num_seeds)
    
    return common_steps, mean_values, std_values, num_seeds
        



if __name__ == '__main__':
    path = '/Users/julianalverio/code/hrl/hindsight_experience_replay/vds_logs/exp631'
    aggregate_success_data(path)
