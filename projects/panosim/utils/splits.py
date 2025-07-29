from typing import Dict, List

# train = \
#     []

# val = \
#     ['scene-001']

test = \
    []

train = \
    ['scene-001', 'scene-002', 'scene-003', 'scene-004', 'scene-005', 'scene-006', 'scene-007',
     'scene-008', 'scene-009', 'scene-010', 'scene-011', 'scene-012', 'scene-013', 'scene-014',
     'scene-015', 'scene-016', 'scene-017', 'scene-018', 'scene-019', 'scene-020', 'scene-021', 
     'scene-022', 'scene-023', 'scene-024', 'scene-025', 'scene-026', 'scene-027', 'scene-028',
     'scene-029', 'scene-030', 'scene-031', 'scene-032', 'scene-033', 'scene-034', 'scene-035',
     'scene-036', 'scene-037', 'scene-038', 'scene-039',]

val = \
    ['scene-040', 'scene-041', 'scene-042', 'scene-043']

def create_splits_scenes(verbose: bool = False) -> Dict[str, List[str]]:
    """
    Similar to create_splits_logs, but returns a mapping from split to scene names, rather than log names.
    The splits are as follows:
    - train/val/test: The standard splits of the nuScenes dataset (700/150/150 scenes).
    - mini_train/mini_val: Train and val splits of the mini subset used for visualization and debugging (8/2 scenes).
    - train_detect/train_track: Two halves of the train split used for separating the training sets of detector and
        tracker if required.
    :param verbose: Whether to print out statistics on a scene level.
    :return: A mapping from split name to a list of scenes names in that split.
    """
    # Use hard-coded splits.
    all_scenes = train + val + test
    # assert len(all_scenes) == 1000 and len(set(all_scenes)) == 1000, 'Error: Splits incomplete!'
    scene_splits = {'train': train, 'val': val, 'test': test}

    # Optional: Print scene-level stats.
    if verbose:
        for split, scenes in scene_splits.items():
            print('%s: %d' % (split, len(scenes)))
            print('%s' % scenes)

    return scene_splits