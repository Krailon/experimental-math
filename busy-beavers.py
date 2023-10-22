from random import randint, choice


class BinaryTuringMachine:
    HALT_STATE = -1

    class HaltedError(RuntimeError): pass

    @staticmethod
    def get_state_table(state_count: int, index: int) -> dict:
        assert type(state_count) is int and state_count >= 0
        assert type(index) is int and index >= 0

        # TODO: fix
        asc = state_count + 1
        state_table = {}
        for in_state in range(state_count):
            for in_bit in (True, False):
                update = (
                    bool(int(index / ((4 * asc) + 4))),
                    #choice((True, False)),  # Output bit
                    int(index / ((2 * asc) + 2)),
                    #choice((-1, 1)),  # Head movement direction
                    (index % asc) - 1
                    #choice((-1, randint(0, state_count - 1)))  # New state
                )
                state_table[in_bit, in_state] = update

        # Debug
        print(F'Generated random state table: {state_table}')

        return state_table

    def __init__(self, state_table: dict, init_state=0):
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
        self.tape = {}  # Sparse binary tape
        self.head_position = 0
        self.age = 0

    def read_tape(self, position=None) -> bool:
        return self.tape.get(position if position is not None else self.head_position, False)

    def write_tape(self, bit: bool):
        assert type(bit) is bool

        if bit:
            self.tape[self.head_position] = bit
        elif self.head_position in self.tape:
            del self.tape[self.head_position]

    def dump_tape(self) -> list:
        indices = sorted(self.tape.keys())
        start = indices[0]
        end = indices[-1]

        return [
            self.read_tape(i)
            for i in range(start, end + 1)
        ]

    def step(self):
        if self.state == BinaryTuringMachine.HALT_STATE:
            raise BinaryTuringMachine.HaltedError()

        # 1) Read tape bit at current head position
        in_bit = self.read_tape()

        # 2) Fetch update parameters from state table based on read tape bit & current state
        out_bit, tape_dir, new_state = self.state_table[in_bit, self.state]

        # 3) Write output bit to tape
        self.write_tape(out_bit)

        # 4) Update head position
        self.head_position += tape_dir

        # 5) Update state
        self.state = new_state

        self.age += 1


class BusyBeaver:
    STEP_LIMIT = int(1e4)

    @staticmethod
    def count_beavers(state_count: int) -> int:
        assert type(state_count) is int and state_count > 0

        return 8 * state_count * (state_count + 1)

    def __init__(self):
        self.beavers = {}  # Binary Turing Machines indexed by their state space size

    def sample_beaver(self, state_count: int, step_limit=None) -> int | None:
        """ Picks a random beaver with the given state count and runs it for the specified step count or until halt. """
        assert type(state_count) is int and state_count >= 0
        assert step_limit is None or (type(step_limit) is int and step_limit > 0)

        if step_limit is None:
            step_limit = BusyBeaver.STEP_LIMIT

        max_beavers = BusyBeaver.count_beavers(state_count)
        if len(self.beavers) >= max_beavers:
            # All beavers of specified order have been sampled so just return one
            beaver_index = choice(list(self.beavers.keys()))
        else:
            beaver_index = None
            while beaver_index is None or (state_count, beaver_index) in self.beavers:
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
                print(F'Beaver [{state_count}:{beaver_index}] was stopped prematurely after step limit of {step_limit}')
                break

        return self.beavers[state_count, beaver_index]


def run():
    n = 2
    bb = BusyBeaver()
    sample_count = 30
    max_1s = 0

    assert sample_count <= BusyBeaver.count_beavers(n)
    for _ in range(sample_count):
        result = bb.sample_beaver(n)
        if result is not None:
            #print(F'Result of randomly sampled beaver with {n} states: {result}')
            if result > max_1s:
                max_1s = result

    print(F'Estimated Σ({n}) = {max_1s} over {sample_count} samples')


if __name__ == '__main__':
    run()
