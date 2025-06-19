import os
import regex as re
import multiprocessing
import time
from collections import Counter
from cs336_basics.pretokenization_example import find_chunk_boundaries

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
    new_dict: dict[tuple[int, ...], int] = Counter()
    for word, count in ori.items():
        if len(word) < 2 or bp not in zip(word[:-1], word[1:]):
            new_dict[word] = count
            continue
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
                bp_freq: dict[tuple[int, int], int] = Counter()
                for tokenid_list, count in word_freq.items():
                    for fir, sec in zip(tokenid_list[:-1], tokenid_list[1:]):
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


def work_one_thread(raw_text: str, sp_tokens: list[str]) -> dict[tuple[int, ...], int]:
    segments = deal_with_sptokens(raw_text, sp_tokens)
    region_freq: dict[tuple[int, ...], int] = Counter()
    for seg in segments:
        if seg in sp_tokens:
            continue
        part_freq = pretoken(seg)
        region_freq.update(part_freq)
    return region_freq


def bpe_parallel(
    input_file: str | os.PathLike, vocab_size: int, sp_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab: dict[int, bytes] = get_init_vocablary(INIT_VOCAB_SIZE, sp_tokens)
    merges: list[tuple[bytes, bytes]] = []
    next_token_id = len(vocab)
    try:
        with open(input_file, "rb") as f:
            cpu_cores: int = os.cpu_count() or 1
            milestone = find_chunk_boundaries(f, cpu_cores, sp_tokens[0].encode("utf-8"))
            # print(f"cpu: {cpu_cores}, regions:{milestone}")
            regions: list[tuple[int, int]] = [(st, ed) for (st, ed) in zip(milestone[:-1], milestone[1:])]

            regions_params = []
            for st, ed in regions:
                f.seek(st)
                chunk_raw_str = f.read(ed - st).decode("utf-8", errors="ignore")
                regions_params.append((chunk_raw_str, sp_tokens))

            with multiprocessing.Pool(processes=len(regions_params)) as pool:
                # print(f"Pretoken with {len(regions_params)} hardware threads")
                result_list = pool.starmap(work_one_thread, regions_params)

            # 合并word-freq
            word_freq: dict[tuple[int, ...], int] = Counter()
            for res in result_list:
                word_freq.update(res)

            # BPE训练循环
            while next_token_id < vocab_size:
                # if next_token_id % 50 == 0:
                #     print(f"Training, size:{next_token_id}")

                bp_freq: dict[tuple[int, int], int] = Counter()
                for tokenid_list, count in word_freq.items():
                    for fir, sec in zip(tokenid_list[:-1], tokenid_list[1:]):
                        bp_freq[fir, sec] += count
                t1 = time.time()
                most_bp = max(bp_freq, key=lambda k: (bp_freq[k], vocab[k[0]], vocab[k[1]]))
                new_bytes = vocab[most_bp[0]] + vocab[most_bp[1]]
                new_id = next_token_id

                vocab[new_id] = new_bytes
                merges.append((vocab[most_bp[0]], vocab[most_bp[1]]))
                next_token_id += 1

                word_freq = merge(word_freq, most_bp, new_id)
        return vocab, merges
    except Exception as e:
        raise RuntimeError(f"Meet Error: {e}")


def bpe_optimized(
    input_file: str | os.PathLike, vocab_size: int, sp_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab: dict[int, bytes] = get_init_vocablary(INIT_VOCAB_SIZE, sp_tokens)
    merges: list[tuple[bytes, bytes]] = []
    next_token_id = len(vocab)

    try:
        with open(input_file, "rb") as f:
            cpu_cores: int = os.cpu_count() or 1
            milestone = find_chunk_boundaries(f, cpu_cores, sp_tokens[0].encode("utf-8"))
            regions: list[tuple[int, int]] = [(st, ed) for (st, ed) in zip(milestone[:-1], milestone[1:])]

            regions_params = []
            for st, ed in regions:
                f.seek(st)
                chunk_raw_str = f.read(ed - st).decode("utf-8", errors="ignore")
                regions_params.append((chunk_raw_str, sp_tokens))

            # 预处理
            with multiprocessing.Pool(processes=len(regions_params)) as pool:
                result_list = pool.starmap(work_one_thread, regions_params)
            word_freq: dict[tuple[int, ...], int] = Counter()
            for res in result_list:
                word_freq.update(res)

            # 初始化字节对频率
            bp_freq: dict[tuple[int, int], int] = Counter()
            for tokenid_list, count in word_freq.items():
                for fir, sec in zip(tokenid_list[:-1], tokenid_list[1:]):
                    bp_freq[fir, sec] += count

            while next_token_id < vocab_size:
                if not bp_freq:
                    break

                most_bp = max(bp_freq, key=lambda k: (bp_freq[k], vocab[k[0]], vocab[k[1]]))
                new_bytes = vocab[most_bp[0]] + vocab[most_bp[1]]
                new_id = next_token_id

                vocab[new_id] = new_bytes
                merges.append((vocab[most_bp[0]], vocab[most_bp[1]]))
                next_token_id += 1

                word_freq, bp_freq = merge_with_incremental_update(word_freq, bp_freq, most_bp, new_id)

        return vocab, merges
    except Exception as e:
        raise RuntimeError(f"Meet Error: {e}")


def merge_with_incremental_update(
    word_freq: dict[tuple[int, ...], int],
    bp_freq: dict[tuple[int, int], int],
    merged_pair: tuple[int, int],
    new_token_id: int,
) -> tuple[dict[tuple[int, ...], int], dict[tuple[int, int], int]]:
    """
    合并字节对并增量更新字节对频率
    """
    affected_words = {}  # 原词 -> (新词, 频次)

    for word, count in word_freq.items():
        if len(word) >= 2:
            # 检查是否包含目标字节对
            has_pair = any((word[i], word[i + 1]) == merged_pair for i in range(len(word) - 1))
            if has_pair:
                new_word = merge_single_word(word, merged_pair, new_token_id)
                affected_words[word] = (new_word, count)

    if not affected_words:
        return word_freq, bp_freq

    # 更新字节对频率
    for old_word, (new_word, count) in affected_words.items():
        # 减去旧词的贡献
        for i in range(len(old_word) - 1):
            pair = (old_word[i], old_word[i + 1])
            bp_freq[pair] -= count
            if bp_freq[pair] <= 0:
                del bp_freq[pair]

    # 更新词频
    new_word_freq = {word: count for word, count in word_freq.items() if word not in affected_words}

    #  更新word_freq
    new_word_counts = Counter()
    for old_word, (new_word, count) in affected_words.items():
        new_word_counts[new_word] += count

    new_word_freq.update(new_word_counts)

    # 添加新词的贡献
    for new_word, count in new_word_counts.items():
        for i in range(len(new_word) - 1):
            pair = (new_word[i], new_word[i + 1])
            bp_freq[pair] = bp_freq.get(pair, 0) + count

    return new_word_freq, bp_freq


def merge_single_word(word: tuple[int, ...], merged_pair: tuple[int, int], new_token_id: int) -> tuple[int, ...]:
    """合并单个词中的字节对"""
    result = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and (word[i], word[i + 1]) == merged_pair:
            result.append(new_token_id)
            i += 2
        else:
            result.append(word[i])
            i += 1
    return tuple(result)
