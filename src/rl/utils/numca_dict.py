import re
from ast import literal_eval


class NumberStateValue:
    def __init__(self, state_getter):
        self._state_getter = state_getter

    @property
    def state_value(self):
        return self._state_getter()


class Numca_dict:
    """
    Compatibility wrapper for the NumCA state-value table.

    A state is the set of unique numbers seen so far in a response. Each response
    contributes its final reward once to every state it passes through.
    """

    def __init__(self, expression_list=None, final_reward=0):
        self.state_stats = {}
        self.root_node = NumberStateValue(lambda: self.state_value(frozenset()))
        if expression_list is not None:
            self.update(expression_list, final_reward)

    def __repr__(self):
        items = []
        for state, stats in sorted(
            self.state_stats.items(),
            key=lambda item: (len(item[0]), sorted(map(str, item[0]))),
        ):
            items.append(
                {
                    "state": sorted(map(str, state)),
                    "count": stats["count"],
                    "reward_sum": stats["reward_sum"],
                }
            )
        return f"NumcaStateTable({items})"

    def __len__(self):
        return len(self.state_stats)

    def _update_state(self, state, reward):
        key = frozenset(state)
        stats = self.state_stats.setdefault(key, {"count": 0.0, "reward_sum": 0.0})
        stats["count"] += 1.0
        stats["reward_sum"] += float(reward)

    def update(self, expression_list, final_reward=0):
        seen_numbers = set()
        self._update_state(seen_numbers, final_reward)
        for number in expression_list:
            if number in seen_numbers:
                continue
            seen_numbers.add(number)
            self._update_state(seen_numbers, final_reward)

    def state_value(self, state):
        stats = self.state_stats.get(frozenset(state))
        if not stats or stats["count"] == 0:
            return 0.0
        return stats["reward_sum"] / stats["count"]

    def advantages(self, expression_list, positions, generation_length, gae_lambda, final_reward=0):
        positions = merge_close_positions(positions, threshold=50)
        grouped_states = []
        seen_numbers = set()

        for number, position in zip(expression_list, positions):
            if number in seen_numbers:
                continue
            seen_numbers.add(number)
            next_state = frozenset(seen_numbers)
            if grouped_states and grouped_states[-1][0] == position:
                grouped_states[-1] = (position, next_state)
            else:
                grouped_states.append((position, next_state))

        advantages = []
        determined_segments = []
        current_state = frozenset()
        last_position = 0

        for position, next_state in grouped_states:
            position = max(0, min(position, generation_length))
            if position > last_position:
                advantage = self.state_value(next_state) - self.state_value(current_state)
                advantages.extend([advantage] * (position - last_position))
                determined_segments.append((last_position, position))
                last_position = position
            current_state = next_state

        if generation_length > last_position:
            advantage = float(final_reward) - self.state_value(current_state)
            advantages.extend([advantage] * (generation_length - last_position))
            determined_segments.append((last_position, generation_length))

        gae_advantage = 0.0
        for start, end in reversed(determined_segments):
            gae_advantage = advantages[start] + (gae_advantage * gae_lambda)
            advantages[start:end] = [gae_advantage] * (end - start)

        if len(advantages) < generation_length:
            advantages.extend([0.0] * (generation_length - len(advantages)))
        elif len(advantages) > generation_length:
            advantages = advantages[:generation_length]
        return advantages


def number_parse(pred: str):
    """
    Parses the numbers in a response.
    """
    number_pattern = re.compile(r"\d+")
    matches = list(number_pattern.finditer(pred))
    numbers = [match.group(0) for match in matches]
    positions = [match.start() for match in matches]
    return numbers[:-1], positions[:-1]


def merge_close_positions(positions: list[int], threshold: int = 50) -> list[int]:
    if not positions:
        return []

    new_positions = list(positions)
    for i in range(len(new_positions) - 2, -1, -1):
        if new_positions[i + 1] - new_positions[i] < threshold:
            new_positions[i] = new_positions[i + 1]

    return new_positions


def Numca_dict_from_string(str_representation: str):
    """
    Parses the current NumCA table representation when available.

    Old serialized graph tables are no longer reconstructed because NumCA no longer
    stores graph links or correct/wrong terminal states.
    """
    if not str_representation:
        return None

    if str_representation.startswith("NumcaStateTable("):
        content = str_representation[len("NumcaStateTable(") : -1]
        try:
            items = literal_eval(content)
        except (ValueError, SyntaxError):
            return Numca_dict()

        table = Numca_dict()
        for item in items:
            state = frozenset(str(number) for number in item.get("state", []))
            table.state_stats[state] = {
                "count": float(item.get("count", 0.0)),
                "reward_sum": float(item.get("reward_sum", 0.0)),
            }
        return table

    return Numca_dict()
