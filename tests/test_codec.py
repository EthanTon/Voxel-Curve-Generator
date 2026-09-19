import unittest

from src.Format.codec import (
    bits_for,
    decode_varints,
    encode_varint,
    encode_varints,
    pack_longs,
    unpack_longs,
)


class BitsTests(unittest.TestCase):
    def test_bits_respect_the_format_floor(self) -> None:
        self.assertEqual(bits_for(1, minimum=4), 4)
        self.assertEqual(bits_for(1, minimum=2), 2)
        self.assertEqual(bits_for(16, minimum=2), 4)
        self.assertEqual(bits_for(17, minimum=2), 5)
        self.assertEqual(bits_for(17, minimum=4), 5)


class PackingTests(unittest.TestCase):
    def test_both_packings_round_trip(self) -> None:
        for bits in (2, 3, 4, 5, 6, 7, 9, 13):
            values = [(index * 7) % (1 << bits) for index in range(4096)]
            for straddle in (False, True):
                with self.subTest(bits=bits, straddle=straddle):
                    packed = pack_longs(values, bits, straddle=straddle)
                    self.assertEqual(
                        unpack_longs(packed, bits, len(values), straddle=straddle),
                        values,
                    )

    def test_a_value_wider_than_its_field_is_masked_not_smeared(self) -> None:
        # An overflow must not land in the next cell's bits.
        packed = pack_longs([0b1111, 0b0001], 2, straddle=False)

        self.assertEqual(unpack_longs(packed, 2, 2, straddle=False), [0b11, 0b01])

    def test_the_two_packings_differ_wherever_bits_do_not_divide_64(self) -> None:
        # 5 bits into 64 leaves 4 spare: anvil pads, litematica does not.
        values = list(range(32))

        padded = pack_longs(values, 5, straddle=False)
        spanning = pack_longs(values, 5, straddle=True)

        self.assertNotEqual(padded, spanning)
        self.assertEqual(len(padded), 3)
        self.assertEqual(len(spanning), 3)
        self.assertEqual(unpack_longs(padded, 5, 32, straddle=False), values)
        self.assertEqual(unpack_longs(spanning, 5, 32, straddle=True), values)

    def test_four_bits_pack_identically_because_they_divide_evenly(self) -> None:
        values = [index % 16 for index in range(64)]

        self.assertEqual(
            pack_longs(values, 4, straddle=False),
            pack_longs(values, 4, straddle=True),
        )

    def test_packed_longs_stay_inside_a_signed_sixty_four_bit_range(self) -> None:
        packed = pack_longs([15] * 32, 4, straddle=False)

        for entry in packed:
            self.assertGreaterEqual(entry, -(2**63))
            self.assertLessEqual(entry, 2**63 - 1)
        self.assertEqual(packed[0], -1)

    def test_a_bit_width_outside_the_word_is_refused(self) -> None:
        for bits in (0, -1, 65):
            with self.assertRaises(ValueError):
                pack_longs([0], bits, straddle=False)


class VarintTests(unittest.TestCase):
    def test_varints_round_trip_past_one_byte(self) -> None:
        values = [0, 1, 127, 128, 255, 300, 16383, 16384, 70000]

        self.assertEqual(decode_varints(encode_varints(values)), values)
        self.assertEqual(encode_varint(128), bytes([0x80, 0x01]))
        self.assertEqual(encode_varint(127), bytes([0x7F]))

    def test_a_negative_index_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            encode_varint(-1)

    def test_a_truncated_varint_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            decode_varints(bytes([0x80]))


if __name__ == "__main__":
    unittest.main()
