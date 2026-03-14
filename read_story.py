"""
Read and print stories from the TinyStories dataset.

Modes:
  random          - Random story from a random shard
  random_from     - Random story from a specific shard (by index)
  specific        - Specific story (by index) from a specific shard (by index)
  list            - Print N stories (or all) from a specific shard

Usage:
  python read_story.py random
  python read_story.py random_from --shard 0
  python read_story.py specific --shard 0 --story 42
  python read_story.py list --shard 0 --count 5
  python read_story.py list --shard 0 --count all
"""

import argparse
import glob
import json
import os
import random
from enum import Enum


class Mode(Enum):
    RANDOM = "random"
    RANDOM_FROM = "random_from"
    SPECIFIC = "specific"
    LIST = "list"


def get_data_dir():
    code_root = os.environ.get("CODE_ROOT_DIR")
    if not code_root:
        raise EnvironmentError(
            "CODE_ROOT_DIR env var not set. "
            "Example: export CODE_ROOT_DIR=/home/jonas"
        )
    return os.path.join(code_root, "Code", "ml", "llama2.c", "data", "TinyStories_all_data")


def get_shard_paths():
    data_dir = get_data_dir()
    shards = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if not shards:
        raise FileNotFoundError(f"No .json files found in {data_dir}")
    return shards


def load_shard(shard_path):
    with open(shard_path, "r") as f:
        return json.load(f)


def print_story(story, shard_idx=None, story_idx=None):
    header_parts = []
    if shard_idx is not None:
        header_parts.append(f"shard={shard_idx}")
    if story_idx is not None:
        header_parts.append(f"story={story_idx}")

    header = f" [{', '.join(header_parts)}]" if header_parts else ""
    print(f"--- Story{header} ---")
    print(story.get("story", "(no story field)").strip())
    print()


def cmd_random():
    shards = get_shard_paths()
    shard_idx = random.randrange(len(shards))
    data = load_shard(shards[shard_idx])
    story_idx = random.randrange(len(data))
    print_story(data[story_idx], shard_idx, story_idx)


def cmd_random_from(shard_idx):
    shards = get_shard_paths()
    if shard_idx < 0 or shard_idx >= len(shards):
        raise IndexError(f"Shard index {shard_idx} out of range (0..{len(shards) - 1})")
    data = load_shard(shards[shard_idx])
    story_idx = random.randrange(len(data))
    print_story(data[story_idx], shard_idx, story_idx)


def cmd_specific(shard_idx, story_idx):
    shards = get_shard_paths()
    if shard_idx < 0 or shard_idx >= len(shards):
        raise IndexError(f"Shard index {shard_idx} out of range (0..{len(shards) - 1})")
    data = load_shard(shards[shard_idx])
    if story_idx < 0 or story_idx >= len(data):
        raise IndexError(f"Story index {story_idx} out of range (0..{len(data) - 1})")
    print_story(data[story_idx], shard_idx, story_idx)


def cmd_list(shard_idx, count):
    shards = get_shard_paths()
    if shard_idx < 0 or shard_idx >= len(shards):
        raise IndexError(f"Shard index {shard_idx} out of range (0..{len(shards) - 1})")
    data = load_shard(shards[shard_idx])

    n = len(data) if count == "all" else min(int(count), len(data))
    print(f"Showing {n} of {len(data)} stories from shard {shard_idx}\n")
    for i in range(n):
        print_story(data[i], shard_idx, i)


def main():
    parser = argparse.ArgumentParser(description="Read stories from TinyStories dataset")
    parser.add_argument("mode", type=str, choices=[m.value for m in Mode],
                        help="random | random_from | specific | list")
    parser.add_argument("--shard", type=int, default=0,
                        help="Shard index (default: 0)")
    parser.add_argument("--story", type=int, default=0,
                        help="Story index within shard (for 'specific' mode)")
    parser.add_argument("--count", type=str, default="5",
                        help="Number of stories to print (for 'list' mode). Use 'all' for everything.")

    args = parser.parse_args()
    mode = Mode(args.mode)

    if mode == Mode.RANDOM:
        cmd_random()
    elif mode == Mode.RANDOM_FROM:
        cmd_random_from(args.shard)
    elif mode == Mode.SPECIFIC:
        cmd_specific(args.shard, args.story)
    elif mode == Mode.LIST:
        cmd_list(args.shard, args.count)


if __name__ == "__main__":
    main()

