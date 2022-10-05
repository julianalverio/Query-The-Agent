import os
import pathlib
import shutil
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--perm', action='store_true')
args = parser.parse_args()

current = pathlib.Path(__file__).parent.resolve()
root = os.path.join(current, 'vds_logs')
root_dirnames = os.listdir(root)
lookup = dict()
for dirname in root_dirnames:
    if not (dirname.startswith('exp') and 'seed' in dirname):
        continue

    split_name = dirname.split('_')
    seed_idx = [idx for idx, x in enumerate(split_name) if 'seed' in x][0]
    split_name.pop(seed_idx)
    no_seed = '_'.join(split_name)
    if no_seed not in lookup:
        lookup[no_seed] = [dirname]
    else:
        lookup[no_seed].append(dirname)
for no_seed, dirnames in lookup.items():
    if args.perm:
        target = os.path.join(root, no_seed)
    else:
        target = os.path.join(root, f'tmp_{no_seed}')
    if not os.path.isdir(target):
        os.mkdir(target)
    for dirname in dirnames:
        source = os.path.join(root, dirname)
        dest = os.path.join(root, target, dirname)
        if not args.perm:
            try:
                shutil.copytree(source, dest)
            # if the file has already been copied before, ignore
            except:
                print('Tried to COPY file, got an error!')
        else:
            try:
                shutil.move(source, dest)
            except:
                print('Tried to MOVE file, got an error!')
             
    

