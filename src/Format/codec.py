from __future__ import annotations

from array import array
from collections.abc import Iterable, Sequence

import numpy as np

WRAP = 1 << 64


def bits_for(palette_size: int, *, minimum: int) -> int:
    # Anvil floors at 4 bits, litematica at 2.
    return max(minimum, max(1, palette_size - 1).bit_length())


def pack_longs(values: Sequence[int], bits: int, *, straddle: bool) -> list[int]:
    return pack_long_array(values, bits, straddle=straddle).tolist()


def pack_long_array(values, bits: int, *, straddle: bool) -> np.ndarray:
    # Signed 64-bit words. Litematica lets an entry span two words (the
    # pre-1.16 layout); anvil pads each word instead.
    if not 1 <= bits <= 64:
        raise ValueError(f"bits must be within 1..64, got {bits}")
    values = np.asarray(values, dtype=np.uint64) & np.uint64((1 << bits) - 1)
    count = values.size
    if straddle:
        start = np.arange(count, dtype=np.uint64) * np.uint64(bits)
        low = (start >> np.uint64(6)).astype(np.intp)
        offset = start & np.uint64(63)
        packed = np.zeros((count * bits + 63) // 64, dtype=np.uint64)
        if count:
            # Each word's entries are contiguous, so OR-reduce them in runs.
            firsts = np.flatnonzero(np.diff(low, prepend=-1))
            packed[low[firsts]] = np.bitwise_or.reduceat(values << offset, firsts)
            spans = ((start + np.uint64(bits - 1)) >> np.uint64(6)).astype(np.intp) != low
            packed[low[spans] + 1] |= values[spans] >> (np.uint64(64) - offset[spans])
    else:
        per_long = 64 // bits
        longs = (count + per_long - 1) // per_long
        padded = np.zeros(longs * per_long, dtype=np.uint64)
        padded[:count] = values
        shifts = np.arange(per_long, dtype=np.uint64) * np.uint64(bits)
        packed = np.bitwise_or.reduce(padded.reshape(longs, per_long) << shifts, axis=1)
    return packed.view(np.int64)


def unpack_longs(packed: Iterable[int], bits: int, count: int, *, straddle: bool) -> list[int]:
    return unpack_long_array(packed, bits, count, straddle=straddle).tolist()


def unpack_long_array(packed, bits: int, count: int, *, straddle: bool) -> np.ndarray:
    if isinstance(packed, np.ndarray):
        words = packed.astype(np.int64).view(np.uint64)
    else:
        words = np.fromiter((int(entry) & (WRAP - 1) for entry in packed), dtype=np.uint64)
    mask = np.uint64((1 << bits) - 1)
    index = np.arange(count, dtype=np.uint64)
    if straddle:
        start = index * np.uint64(bits)
        low = (start >> np.uint64(6)).astype(np.intp)
        offset = start & np.uint64(63)
        values = words[low] >> offset
        high = ((start + np.uint64(bits - 1)) >> np.uint64(6)).astype(np.intp)
        spans = high != low
        values[spans] |= words[high[spans]] << (np.uint64(64) - offset[spans])
    else:
        per_long = np.uint64(64 // bits)
        values = words[(index // per_long).astype(np.intp)] >> (index % per_long * np.uint64(bits))
    return values & mask


def encode_varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("palette indices are never negative")
    out = bytearray()
    while True:
        chunk = value & 0x7F
        value >>= 7
        if not value:
            out.append(chunk)
            return bytes(out)
        out.append(chunk | 0x80)


def encode_varints(values: Iterable[int]) -> bytes:
    if isinstance(values, (array, memoryview)):
        return _encode_varint_array(np.frombuffer(values, dtype=np.uint32))
    return b"".join(encode_varint(value) for value in values)


def _encode_varint_array(values: np.ndarray) -> bytes:
    if values.size == 0:
        return b""
    largest = int(values.max())
    if largest < 0x80:
        return values.astype(np.uint8).tobytes()
    width = np.ones(values.shape, dtype=np.uint8)
    for bits in (7, 14, 21, 28):
        if largest < 1 << bits:
            break
        width += values >= (1 << bits)
    ends = np.cumsum(width, dtype=np.int64)
    starts = ends - width
    out = np.empty(int(ends[-1]), dtype=np.uint8)
    for byte in range(int(width.max())):
        here = width > byte
        chunk = (values[here] >> (7 * byte)) & 0x7F
        more = (width[here] > byte + 1).astype(np.uint32) << 7
        out[starts[here] + byte] = chunk | more
    return out.tobytes()


def decode_varints_array(data: bytes | bytearray) -> np.ndarray:
    raw = np.frombuffer(bytes(data), dtype=np.uint8)
    if raw.size == 0:
        return np.zeros(0, dtype=np.int64)
    last = (raw & 0x80) == 0
    if not last[-1]:
        raise ValueError("varint stream ends mid-value")
    ends = np.flatnonzero(last)
    starts = np.concatenate(([0], ends[:-1] + 1))
    width = ends - starts + 1
    values = np.zeros(ends.size, dtype=np.int64)
    for byte in range(int(width.max())):
        here = width > byte
        values[here] |= (raw[starts[here] + byte].astype(np.int64) & 0x7F) << (7 * byte)
    return values


def decode_varints(data: bytes | bytearray) -> list[int]:
    return decode_varints_array(data).tolist()

