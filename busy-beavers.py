from math import log2
from random import randint, choice
from yaml import safe_dump as dump_yaml, safe_load as load_yaml


class BinaryTuringMachine:
    HALT_STATE = -1

    class HaltedError(RuntimeError): pass

    @staticmethod
    def get_state_table(state_count: int, index: int) -> dict[tuple[bool, int], tuple[bool, int, int]]:
        assert type(state_count) is int and state_count >= 0
        assert type(index) is int and index >= 0

        # Note: this encoding scheme has holes since any state count which isn't a power of 2 will waste encoding bits
        # TODO: consider expanding "instructions" to include third head movement option 0 (would add n-bit waste)
        full_state_count = state_count + 1  # Special Halt state not included in state count
        state_bit_count = log2(full_state_count)  # Number of bits needed to encode all states, including Halt
        bits = bin(index)[2:].rjust(state_count * 6, '0')  # Bit string decoding of index
        bits_per_state = 2 * (2 + state_bit_count)  # Each state's column uses variable-bit count based on state count
        bit_counts = [1, 1, state_bit_count]  # Each cell uses 1 bit Output, 1 bit Direction, variable-bit state count

        # Populate vector
        vector = []
        for si in range(state_count):
            for bi in range(2):
                state_offset = (si * bits_per_state) + bi
                field_offset = 0

                for count in bit_counts:
                    vector = [int(bits[state_offset + field_offset:][:count], 2)] + vector
                    field_offset += count

        state_table = {}
        for in_state in range(state_count):
            for in_bit in (True, False):
                vector_offset = in_state * 6
                out_bit = bool(vector[vector_offset])
                move_dir: int = (-1, 1)[vector[vector_offset + 1]]
                new_state = vector[vector_offset + 2] % full_state_count

                update = (out_bit, move_dir, new_state)
                state_table[in_bit, in_state] = update

        # Debug
        print(F'Generated state table for index {index}: {state_table}')

        return state_table

    def __init__(self, state_table: dict[tuple[bool, int], tuple[bool, int, int]], init_state=0):
        assert type(state_table) is dict
        assert {type(k) for k in state_table.keys()} == {tuple}
        assert {len(k) for k in state_table.keys()} == {2}
        assert {(type(k[0]), type(k[1])) for k in state_table.keys()} == {(bool, int)}
        assert {type(v) for v in state_table.values()} == {tuple}
        assert {len(v) for v in state_table.values()} == {3}
        assert {(type(v[0]), type(v[1]), type(v[2])) for v in state_table.values()} == {(bool, int, int)}
        assert type(init_state) is int and init_state >= 0

        self.state_table = state_table
        self.state = init_state
        self.tape: dict[int, bool] = {}  # Sparse binary tape
        self.head_position = 0
        self.age = 0

    def read_tape(self, position=None, as_int=False) -> bool | int:
        value = self.tape.get(position if position is not None else self.head_position, False)

        return int(value) if as_int else value

    def write_tape(self, bit: bool):
        assert type(bit) is bool

        if bit:
            self.tape[self.head_position] = bit
        elif self.head_position in self.tape:
            del self.tape[self.head_position]

    def dump_tape(self, as_ints=False) -> list[bool] | list[int]:
        indices = sorted(self.tape.keys())
        start = indices[0]
        end = indices[-1]

        return [
            self.read_tape(i, as_ints)
            for i in range(start, end + 1)
        ]

    def step(self):
        if self.state == BinaryTuringMachine.HALT_STATE:
            raise BinaryTuringMachine.HaltedError()

        # 1) Read tape bit at current head position
        in_bit = self.read_tape()

        # 2) Fetch update parameters from state table based on read tape bit & current state
        out_bit, tape_dir, new_state = self.state_table[in_bit, self.state]
        assert type(out_bit) is bool, \
            F'Loaded invalid output bit from state table @ ({in_bit}, {self.state}): {out_bit}'
        assert tape_dir in (-1, 1), \
            F'Loaded invalid tape direction from state table @ ({in_bit}, {self.state}): {tape_dir}'
        assert new_state == -1 or 0 <= new_state < (len(self.state_table) // 2), \
            F'Loaded invalid new state from state table @ ({in_bit}, {self.state}): {new_state}'

        # 3) Write output bit to tape
        self.write_tape(out_bit)

        # 4) Update head position
        self.head_position += tape_dir

        # 5) Update state
        self.state = new_state

        self.age += 1


class BusyBeaver:
    STEP_LIMIT = int(1e5)

    @staticmethod
    def count_beavers(state_count: int) -> int:
        assert type(state_count) is int and state_count > 0

        # Given n states:
        # - For each input bit there are b = 2 * 2 * (n + 1) = 4(n + 1) = 4n + 4 combinations
        # - For each input state there are b^2 = (4n + 4)^2 = 16n^2 + 32n + 16 combinations
        # - For each table there are (b^2)^n = b^2+n combinations

        per_bit = (4 * state_count) + 4
        per_state = per_bit**2

        return per_state**state_count

    def __init__(self):
        self.beavers = {}  # Binary Turing Machines indexed by their state space size
        # TODO: load saved beavers from YAML

    def sample_beaver(self, state_count: int, step_limit=None, new=False) -> int | None:
        """ Picks a random beaver with the given state count and runs it for the specified step count or until halt. """
        assert type(state_count) is int and state_count >= 0
        assert step_limit is None or (type(step_limit) is int and step_limit > 0)
        assert type(new) is bool

        if step_limit is None:
            step_limit = BusyBeaver.STEP_LIMIT

        max_beavers = BusyBeaver.count_beavers(state_count)
        if len(self.beavers) >= max_beavers:
            # All beavers of specified order have been sampled so just return one
            beaver_index = choice(list(self.beavers.keys()))
        else:
            beaver_index = None
            while beaver_index is None or (new and (state_count, beaver_index) in self.beavers):
                beaver_index = randint(0, max_beavers - 1)

        state_table = BinaryTuringMachine.get_state_table(state_count, beaver_index)
        beaver = BinaryTuringMachine(state_table)

        # Run beaver until halting, for specified number of steps or until reaching system step limit
        while True:
            try:
                beaver.step()
            except BinaryTuringMachine.HaltedError:
                tape_1s = len(beaver.tape)
                self.beavers[state_count, beaver_index] = tape_1s
                print(F'Beaver [{state_count}:{beaver_index}] halted after {beaver.age} steps yielding {tape_1s} 1s')
                break

            if step_limit is not None and beaver.age >= step_limit:
                self.beavers[state_count, beaver_index] = None
                #print(F'Beaver [{state_count}:{beaver_index}] was stopped prematurely after step limit of {step_limit}')
                break

        # TODO: dump beavers to YAML

        return self.beavers[state_count, beaver_index]


def generate_beavers(beaver_count: int, state_count=1, verbose=True) -> list[tuple[bool, int, int, bool, int, int]]:
    assert type(beaver_count) is int and beaver_count > 0
    assert type(state_count) is int and state_count > 0

    assert state_count == 1  # Debug

    bits = (False, True)
    move_dirs = (-1, 1)
    new_states = (-1,) + tuple(range(1, state_count + 1))
    vector = [0] * 6  # TODO: extend to arbitrary state counts

    def increment_vector():
        i = 1
        while True:
            v_range = (len(new_states), len(move_dirs), len(bits))[i % 3]
            vector[-i] = (vector[-i] + 1) % v_range
            if vector[-i] > 0:
                break
            elif i < len(vector):
                i += 1
            else:
                break

    def get_table() -> tuple[bool, int, int, bool, int, int]:
        # TODO: extend to arbitrary state counts
        return (
            bits[vector[0]], move_dirs[vector[1]], new_states[vector[2]],
            bits[vector[3]], move_dirs[vector[4]], new_states[vector[5]]
        )

    cache = set()
    tables = []
    beaver_count = min(beaver_count, BusyBeaver.count_beavers(state_count))
    for b in range(beaver_count):
        table = get_table()
        assert table not in cache  # Prevent unintended duplicates

        if verbose:
            print(F'[{b}] {vector}: {table}')

        cache.add(table)
        tables.append(table)
        increment_vector()

    return tables


def validate_beaver_generation(state_count=1):
    assert type(state_count) is int and state_count > 0

    beaver_count = BusyBeaver.count_beavers(state_count)
    beavers = generate_beavers(beaver_count, state_count, verbose=False)

    for index, beaver in enumerate(beavers):
        beaver_by_index = BinaryTuringMachine.get_state_table(state_count, index)
        assert {0: beaver} == beaver_by_index, F'Beaver {index} failed validation: {beaver} != {beaver_by_index}'


def run(unique=False):
    n = 1
    bb = BusyBeaver()
    sample_count = 64  # Too many for n=1 but it should compensate
    max_1s = 0

    assert sample_count <= BusyBeaver.count_beavers(n)
    for _ in range(sample_count):
        result = bb.sample_beaver(n, new=unique)
        if result is not None:
            #print(F'Result of randomly sampled beaver with {n} states: {result}')
            if result > max_1s:
                max_1s = result

    print(F'Estimated Σ({n}) = {max_1s} over {sample_count} samples')


if __name__ == '__main__':
    #run(True)
    #s1_beavers = generate_beavers(64)
    validate_beaver_generation()
