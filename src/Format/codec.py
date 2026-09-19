from __future__ import annotations

from collections.abc import Iterable, Sequence

SIGN_BIT = 1 << 63
WRAP = 1 << 64


def bits_for(palette_size: int, *, minimum: int) -> int:
    # Anvil floors at 4 bits, litematica at 2.
    return max(minimum, max(1, palette_size - 1).bit_length())


def pack_longs(values: Sequence[int], bits: int, *, straddle: bool) -> list[int]:
    # Litematica keeps the pre-1.16 layout where an entry may span two longs;
    # anvil since 1.16 pads each long instead. Swapping them writes a file
    # that loads and is quietly wrong.
    if not 1 <= bits <= 64:
        raise ValueError(f"bits must be within 1..64, got {bits}")
    mask = (1 << bits) - 1
    if straddle:
        packed = [0] * ((len(values) * bits + 63) // 64)
        for index, value in enumerate(values):
            start = index * bits
            low, offset = start >> 6, start & 63
            packed[low] |= (value & mask) << offset
            high = (start + bits - 1) >> 6
            if high != low:
                packed[high] |= (value & mask) >> (64 - offset)
    else:
        per_long = 64 // bits
        packed = [0] * ((len(values) + per_long - 1) // per_long)
        for index, value in enumerate(values):
            packed[index // per_long] |= (value & mask) << (index % per_long * bits)
    return [_signed(entry & (WRAP - 1)) for entry in packed]


def unpack_longs(packed: Iterable[int], bits: int, count: int, *, straddle: bool) -> list[int]:
    # int() first: numpy int64 entries off a LongArrayTag cannot hold the mask.
    entries = [int(entry) & (WRAP - 1) for entry in packed]
    mask = (1 << bits) - 1
    values = []
    if straddle:
        for index in range(count):
            start = index * bits
            low, offset = start >> 6, start & 63
            value = entries[low] >> offset
            high = (start + bits - 1) >> 6
            if high != low:
                value |= entries[high] << (64 - offset)
            values.append(value & mask)
    else:
        per_long = 64 // bits
        for index in range(count):
            entry = entries[index // per_long]
            values.append((entry >> (index % per_long * bits)) & mask)
    return values


def encode_varint(value: int) -> bytes:
    # .schem block data is a varint stream inside a byte array.
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
    return b"".join(encode_varint(value) for value in values)


def decode_varints(data: bytes | bytearray) -> list[int]:
    out, value, shift, partial = [], 0, 0, False
    for raw in bytes(data):
        partial = True
        value |= (raw & 0x7F) << shift
        if raw & 0x80:
            shift += 7
            continue
        out.append(value)
        value, shift, partial = 0, 0, False
    if partial:
        raise ValueError("varint stream ends mid-value")
    return out


def _signed(value: int) -> int:
    return value - WRAP if value & SIGN_BIT else value
