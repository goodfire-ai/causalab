"""Optional values and fixed bounded steps, runnable in files or notebook cells."""

from causalab.causal import Dom, V, family, mechanism, require, submodel
from causalab.causal.model import CausalModel


@mechanism
def optional_update(
    enabled: Dom([False, True]),
    x: Dom(range(10)),
    base: Dom(range(20)),
):
    candidate = V(x + 1 if enabled else None)
    result = V(base if candidate is None else candidate)
    raw_input = V(f"{enabled}:{x}:{base}", domain=Dom(str))  # noqa: F841
    raw_output = V(str(result), domain=Dom(str))  # noqa: F841
    return result


def make_bounded_countdown(max_steps=4):
    if max_steps < 0:
        raise ValueError("max_steps must be nonnegative")
    # The extra input value exercises a computation that exceeds the step bound.
    State = Dom(range(max_steps + 2))

    @submodel
    def run(initial):
        state = family(size=max_steps + 1, domain=State)
        running = family(size=max_steps + 1)
        state[0] = initial
        running[0] = state[0] > 0

        for t in range(max_steps):
            if running[t]:
                state[t + 1] = max(0, state[t] - 1)
            else:
                state[t + 1] = state[t]
            running[t + 1] = running[t] and state[t + 1] > 0

        require(not running[max_steps], error="step bound exceeded")
        final = V(state[max_steps])
        return final

    @mechanism
    def bounded_countdown(initial: State):
        countdown = run(initial)
        raw_input = V(str(initial), domain=Dom(str))  # noqa: F841
        raw_output = V(str(countdown), domain=Dom(str))  # noqa: F841
        return countdown

    return CausalModel(bounded_countdown, id="bounded_countdown")


OPTIONAL_UPDATE = CausalModel(optional_update, id="optional_update")
