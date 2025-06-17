import os
import regex as re
from collections import defaultdict, Counter

INIT_VOCAB_SIZE = 256


def get_init_vocablary(init_size: int, sp_tokens: list[str]) -> dict[int, bytes]:
    """
    Returns the initial vocabulary for the BPE tokenizer.
    """
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(0, init_size)}
    for i, token in enumerate(sp_tokens):
        vocab[init_size + i] = token.encode("utf-8", errors="ignore")
    return vocab


def deal_with_sptokens(text: str, sp_tokens: list[str]) -> list[str]:
    """
    Deal with the special tokens

    Args:
        text (str): origin text
        sp_tokens (list[str]): special tokens
    """
    sp_tokens = sorted(sp_tokens, key=lambda x: -len(x))
    SP_SPLIT_PAT = "(" + "".join(re.escape(tok) for tok in sp_tokens) + ")"
    parts = re.split(SP_SPLIT_PAT, text)
    return parts


def bytes2tuple_of_int(bts: bytes) -> tuple[int, ...]:
    return tuple(bts)


def pretoken(text: str) -> dict[tuple[int, ...], int]:
    """
    get the word-frequency table of a text

    Args:
        text(str): input text, should has been handled with sp-tokens, should not passed in a sp-token
    Returns:
        dict[str, int]: word and its frequency
    """
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    words = [bytes2tuple_of_int(match.group().encode("utf-8")) for match in re.finditer(PAT, text)]
    return Counter(words)


def merge(ori: dict[tuple[int, ...], int], bp: tuple[int, int], new_tokenid: int) -> dict[tuple[int, ...], int]:
    new_dict: dict[tuple[int, ...], int] = defaultdict(int)
    for word, count in ori.items():
        new_represent = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == bp:
                new_represent.append(new_tokenid)
                i += 2
            else:
                new_represent.append(word[i])
                i += 1
        new_dict[tuple(new_represent)] = count
    return new_dict


def bpe_single_thread(
    input_file: str | os.PathLike, vocab_size: int, sp_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab: dict[int, bytes] = get_init_vocablary(INIT_VOCAB_SIZE, sp_tokens)
    merges: list[tuple[bytes, bytes]] = []
    next_token_id = len(vocab)

    try:
        with open(input_file) as f:
            raw_text: str = f.read()
            handled_segments = deal_with_sptokens(raw_text, sp_tokens)
            word_freq: dict[tuple[int, ...], int] = Counter()

            # get init word-freq
            for seg in handled_segments:
                if seg in sp_tokens:
                    continue

                part_of_total = pretoken(seg)
                word_freq.update(part_of_total)

            while next_token_id < vocab_size:
                if next_token_id % 10 == 0:
                    print(f"Training, size: {next_token_id}")
                # get the bp-freq
                bp_freq: dict[tuple[int, int], int] = defaultdict(int)
                for tokenid_list, count in word_freq.items():
                    for fir, sec in zip(tokenid_list, tokenid_list[1:]):
                        bp_freq[fir, sec] += count

                # get most bytes-pair
                most_bp = max(bp_freq, key=lambda k: (bp_freq[k], vocab[k[0]], vocab[k[1]]))
                new_bytes = vocab[most_bp[0]] + vocab[most_bp[1]]
                new_id = next_token_id

                vocab[new_id] = new_bytes
                merges.append((vocab[most_bp[0]], vocab[most_bp[1]]))
                next_token_id += 1

                word_freq = merge(word_freq, most_bp, new_id)

    except Exception as e:
        raise RuntimeError(f"Meet Error: {e}")
    return vocab, merges


def test():
    pass


if __name__ == "__main__":
    test()
