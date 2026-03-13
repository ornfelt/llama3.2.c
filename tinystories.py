"""
Download, preprocess and serve the TinyStories dataset as a DataLoader.
"""

import argparse
import glob
import json
import os
import random
from typing import List
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import requests
import sentencepiece as spm
import torch
import torch.distributed as dist
from tqdm import tqdm

from tokenizer import Tokenizer

DATA_CACHE_DIR = "data"

# Llama 3 BOS/EOS token IDs
LLAMA3_BOS_ID = 128000  # <|begin_of_text|>
LLAMA3_EOS_ID = 128001  # <|end_of_text|>

def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def download():
    """Downloads the TinyStories dataset to DATA_CACHE_DIR"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # download the TinyStories dataset, unless it's already downloaded
    data_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
    data_filename = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data.tar.gz")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
    else:
        print(f"{data_filename} already exists, skipping download...")

    # unpack the tar.gz file into all the data shards (json files)
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(f"Unpacking {data_filename}...")
        os.system(f"tar -xzf {data_filename} -C {data_dir}")
    else:
        print(f"{data_dir} already exists, skipping unpacking...")

    # print a single example just for debugging and such
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    with open(shard_filenames[0], "r") as f:
        data = json.load(f)
    print("Download done.")
    print(f"Number of shards: {len(shard_filenames)}")
    print(f"Example story:\n{data[0]}")

def train_vocab(vocab_size):
    """
    Trains a custom sentencepiece tokenizer on the TinyStories dataset.
    The custom tokenizer files will be saved in DATA_CACHE_DIR/tok{N} directories,
    where N is the vocab size. This is also where the pretok .bin files will go.
    """
    assert vocab_size > 0, "Vocab size must be positive"

    # output file prefix path for sentencepiece
    prefix = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")

    # how many shards we'll use for vocab training, kept low for efficiency
    num_shards = 10

    # 1) export a large chunk of text as a single text file tiny.txt
    tiny_file = os.path.join(DATA_CACHE_DIR, "tiny.txt")
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    print(f"Writing temporary file {tiny_file} with {num_shards} shards...")
    with open(tiny_file, "w", encoding="utf-8") as of:
        for shard in tqdm(shard_filenames[:num_shards]):
            with open(shard, "r") as f:
                data = json.load(f)
            for example in data:
                text = example["story"]
                text = text.strip()
                of.write(text + "\n")
    print(f"Size is: {os.path.getsize(tiny_file) / 1024 / 1024:.2f} MB")

    # 2) train the sentencepiece model
    print("Will now train the vocab...")
    spm.SentencePieceTrainer.train(input=tiny_file,
                                   model_prefix=prefix,
                                   model_type="bpe",
                                   vocab_size=vocab_size,
                                   self_test_sample_size=0,
                                   input_format="text",
                                   character_coverage=1.0,
                                   num_threads=os.cpu_count(),
                                   split_digits=True,
                                   allow_whitespace_only_pieces=True,
                                   byte_fallback=True,
                                   unk_surface=r" \342\201\207 ",
                                   normalization_rule_name="identity")

    # 3) optional cleanup, ask the user if they'd like to delete tiny.txt
    dec = input(f"Delete the temporary file {tiny_file}? [y/N] ")
    if dec.lower() == "y":
        os.remove(tiny_file)
        print(f"Deleted {tiny_file}")

    print(f"Trained tokenizer is in {prefix}.model")
    print("Done.")


def _get_llama3_tokenizer():
    """
    Returns a tiktoken-based Llama 3 tokenizer.
    Requires the tokenizer.model file from Meta's Llama 3 release.
    We look for it at DATA_CACHE_DIR/llama3_tokenizer.model or the path
    set via the LLAMA3_TOKENIZER_PATH env var.
    """
    try:
        import tiktoken
        from tiktoken.load import load_tiktoken_bpe
    except ImportError:
        raise ImportError(
            "tiktoken is required for Llama 3 tokenization. "
            "Install it with: pip install tiktoken"
        )

    tokenizer_path = os.environ.get(
        "LLAMA3_TOKENIZER_PATH",
        os.path.join(DATA_CACHE_DIR, "llama3_tokenizer.model")
    )
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"Llama 3 tokenizer not found at {tokenizer_path}. "
            f"Download it from Meta's Llama 3 release and place it there, "
            f"or set LLAMA3_TOKENIZER_PATH env var."
        )

    mergeable_ranks = load_tiktoken_bpe(tokenizer_path)

    # Special tokens for Llama 3
    num_base_tokens = len(mergeable_ranks)
    special_tokens = {
        "<|begin_of_text|>": 128000,
        "<|end_of_text|>": 128001,
        "<|start_header_id|>": 128006,
        "<|end_header_id|>": 128007,
        "<|eot_id|>": 128009,
    }
    # Reserved special tokens 0-250
    for i in range(256):
        token = f"<|reserved_special_token_{i}|>"
        if token not in special_tokens:
            special_tokens[token] = 128002 + i  # fills gaps from 128002

    enc = tiktoken.Encoding(
        name="llama3",
        pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )
    return enc


class Llama3Tokenizer:
    """Thin wrapper around tiktoken to match the Tokenizer interface used here."""

    def __init__(self):
        self.enc = _get_llama3_tokenizer()
        self.bos_id = LLAMA3_BOS_ID
        self.eos_id = LLAMA3_EOS_ID

    def encode(self, text, bos=False, eos=False):
        tokens = self.enc.encode(text, allowed_special=set())
        if bos:
            tokens = [self.bos_id] + tokens
        if eos:
            tokens = tokens + [self.eos_id]
        return tokens


def process_shard(args, vocab_size, vocab_source="llama2"):
    shard_id, shard = args

    if vocab_source == "llama3":
        enc = Llama3Tokenizer()
        bos_id = LLAMA3_BOS_ID
        token_dtype = np.uint32  # llama3 token IDs exceed uint16 range
    else:
        tokenizer_model = get_tokenizer_model_path(vocab_size)
        enc = Tokenizer(tokenizer_model)
        bos_id = 1
        token_dtype = np.uint16

    with open(shard, "r") as f:
        data = json.load(f)
    all_tokens = []
    for example in tqdm(data, position=shard_id):
        text = example["story"]
        text = text.strip()  # get rid of leading/trailing whitespace
        tokens = enc.encode(text, bos=True, eos=False)  # encode the text, use BOS
        all_tokens.extend(tokens)
    # convert to nparray
    all_tokens = np.array(all_tokens, dtype=token_dtype)
    # calculate the output filename
    if vocab_source == "llama3":
        # save .bin files into a llama3_tok directory
        bin_dir = os.path.join(DATA_CACHE_DIR, "llama3_tok")
        shard_basename = os.path.basename(shard)
        bin_basename = shard_basename.replace(".json", ".bin")
        tokenized_filename = os.path.join(bin_dir, bin_basename)
    elif vocab_size == 0:
        # if we're using Llama 2, just save the tokenized file in the same dir
        tokenized_filename = shard.replace(".json", ".bin")
    else:
        # save .bin files into a new tok{N} directory
        bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
        shard_basename = os.path.basename(shard)
        bin_basename = shard_basename.replace(".json", ".bin")
        tokenized_filename = os.path.join(bin_dir, bin_basename)
    # write the bytes
    with open(tokenized_filename, "wb") as f:
        f.write(all_tokens.tobytes())
    # calculate the average sequence length (they are separated by BOS token)
    avg_seq_len = all_tokens.size / ((all_tokens == bos_id).sum())
    print(f"Saved {tokenized_filename}, average seqlen: {avg_seq_len:.2f}")


def pretokenize(vocab_size, vocab_source="llama2", max_shards=0):
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if len(shard_filenames) == 0:
        print(f"ERROR: No .json shard files found in {data_dir}")
        print("Did you run 'python tinystories.py download' first?")
        return

    # Limit shards if requested
    if max_shards > 0 and max_shards < len(shard_filenames):
        print(f"Limiting to {max_shards} of {len(shard_filenames)} shards (--max_shards={max_shards})")
        shard_filenames = shard_filenames[:max_shards]

    print(f"Found {len(shard_filenames)} shards to tokenize with vocab_source={vocab_source}")

    if vocab_source == "llama3":
        bin_dir = os.path.join(DATA_CACHE_DIR, "llama3_tok")
        os.makedirs(bin_dir, exist_ok=True)
    elif vocab_size > 0:
        # .bin files will be saved into tok{N} directory, create it once here
        bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
        os.makedirs(bin_dir, exist_ok=True)

    # process all the shards in a process pool
    fun = partial(process_shard, vocab_size=vocab_size, vocab_source=vocab_source)
    with ProcessPoolExecutor() as executor:
        list(executor.map(fun, enumerate(shard_filenames)))  # list() forces execution & surfaces errors
    print("Done.")


class PretokDataset(torch.utils.data.IterableDataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(self, split, max_seq_len, vocab_size, vocab_source):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.vocab_source = vocab_source

    def __iter__(self):
        # get worker info within a DataLoader
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        # get DDP rank info
        rank = dist.get_rank() if dist.is_initialized() else 0
        # combine the worker_id and worker_rank to create a unique seed for rng
        seed = 42 + worker_id + 1337 * rank
        rng = random.Random(seed)
        print(f"Created a PretokDataset with rng seed {seed}")

        # determine bin directory and dtype based on vocab source
        if self.vocab_source == "llama3":
            bin_dir = os.path.join(DATA_CACHE_DIR, "llama3_tok")
            token_dtype = np.uint32
        elif self.vocab_source == "llama2":
            # the .bin files are right along the .json files
            bin_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
            token_dtype = np.uint16
        elif self.vocab_source == "custom":
            # the .bin files are in tok{N} directory
            bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{self.vocab_size}")
            token_dtype = np.uint16
        else:
            raise ValueError(f"Unknown vocab_source: {self.vocab_source}")

        shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))
        # train/test split. let's use only shard 0 for test split, rest train
        shard_filenames = shard_filenames[1:] if self.split == "train" else shard_filenames[:1]
        assert len(shard_filenames)>0, f"No bin files found in {bin_dir}"
        while True:
            rng.shuffle(shard_filenames)
            for shard in shard_filenames:
                # open the dataset for reading but keep it on disk with memmap
                m = np.memmap(shard, dtype=token_dtype, mode="r")
                num_batches = len(m) // self.max_seq_len
                num_batches -= 1  # drop the last partial batch
                assert num_batches > 0, "this shard is way too small? investigate."
                ixs = list(range(num_batches))
                rng.shuffle(ixs)
                for ix in ixs:
                    start = ix * self.max_seq_len
                    end = start + self.max_seq_len + 1
                    # calling .astype will copy the data into a new numpy array, now in RAM
                    chunk = torch.from_numpy((m[start:end]).astype(np.int64))
                    x = chunk[:-1]
                    y = chunk[1:]
                    yield x, y

# -----------------------------------------------------------------------------
# public interface functions

def get_tokenizer_model_path(vocab_size):
    """
    Returns path to the sentencepiece tokenizer model for a given vocab size
    vocab_size = 0 designates the default Llama 2 tokenizer, in that case
    None is returned.
    """
    if vocab_size == 0:
        return None
    else:
        return os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}.model")

class Task:

    @staticmethod
    def iter_batches(batch_size, device, num_workers=0, **dataset_kwargs):
        ds = PretokDataset(**dataset_kwargs)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y

# -----------------------------------------------------------------------------
# CLI for constructing the dataset

if __name__ == "__main__":
    """
    These stages are designed to be run in order.

    To tokenize data with the Llama 2 tokenizer:
    python tinystories.py download
    python tinystories.py pretokenize

    To tokenize data with the Llama 3 tokenizer:
    python tinystories.py download
    python tinystories.py pretokenize --vocab_source=llama3

    To tokenize data with a custom tokenizer we train ourselves with sentencepiece, e.g.:
    python tinystories.py download
    python tinystories.py train_vocab --vocab_size=2048
    python tinystories.py pretokenize --vocab_size=2048
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=["download", "pretokenize", "train_vocab"])
    parser.add_argument("--vocab_size", type=int, default=0, help="pretokenization vocab size. 0 = use Llama 2 tokenizer.")
    parser.add_argument("--vocab_source", type=str, default="llama2", choices=["llama2", "llama3", "custom"],
                        help="tokenizer to use: llama2, llama3, or custom")
    parser.add_argument("--max_shards", type=int, default=0, help="only tokenize first N shards (0 = all)")
    args = parser.parse_args()

    # depending on the stage call the appropriate function
    if args.stage == "download":
        download()
    elif args.stage == "train_vocab":
        train_vocab(vocab_size=args.vocab_size)
    elif args.stage == "pretokenize":
        pretokenize(vocab_size=args.vocab_size, vocab_source=args.vocab_source, max_shards=args.max_shards)
    else:
        raise ValueError(f"Unknown stage {args.stage}")
