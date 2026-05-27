from __future__ import annotations

from typing import Dict, Iterable, List


OPTION_ORDER = ["A", "B", "C", "D", "E"]
# Paper formula: w_m(k) = 4 - d_m(k), where d_m(k) is the
# option-axis distance between option k and the stage-1 majority m.
MAJORITY_ALIGNMENT_WEIGHTS = {
    "A": {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0, "E": 0.0},
    "B": {"A": 3.0, "B": 4.0, "C": 3.0, "D": 2.0, "E": 1.0},
    "C": {"A": 2.0, "B": 3.0, "C": 4.0, "D": 3.0, "E": 2.0},
    "D": {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 3.0},
    "E": {"A": 0.0, "B": 1.0, "C": 2.0, "D": 3.0, "E": 4.0},
}


def majority_option(choices: Dict[str, str]) -> str:
    counts = {option: 0 for option in OPTION_ORDER}
    for choice in choices.values():
        counts[choice] += 1
    return max(OPTION_ORDER, key=lambda option: (counts[option], -OPTION_ORDER.index(option)))


def option_counts(choices: Dict[str, str]) -> Dict[str, int]:
    counts = {option: 0 for option in OPTION_ORDER}
    for choice in choices.values():
        counts[choice] += 1
    return counts


def mai(choice: str, majority: str) -> float:
    return MAJORITY_ALIGNMENT_WEIGHTS[majority][choice]


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def minority_agents(stage1_choices: Dict[str, str], majority: str) -> List[str]:
    return [agent_id for agent_id, choice in stage1_choices.items() if choice != majority]


def agent_mai_table(
    stage1_choices: Dict[str, str],
    stage2_choices: Dict[str, str],
    majority: str,
    minority: List[str] | None = None,
) -> Dict[str, Dict[str, float]]:
    minority = minority if minority is not None else minority_agents(stage1_choices, majority)
    return {
        agent_id: {
            "stage1_choice": stage1_choice,
            "stage2_choice": stage2_choices[agent_id],
            "is_stage1_minority": agent_id in minority,
            "MAI_stage1": mai(stage1_choice, majority),
            "MAI_stage2": mai(stage2_choices[agent_id], majority),
            "delta_MAI": mai(stage2_choices[agent_id], majority) - mai(stage1_choice, majority),
        }
        for agent_id, stage1_choice in stage1_choices.items()
    }


def minority_delta_mai(
    stage1_choices: Dict[str, str],
    stage2_choices: Dict[str, str],
    majority: str,
) -> float:
    agents = minority_agents(stage1_choices, majority)
    deltas = []
    for agent_id in agents:
        before = mai(stage1_choices[agent_id], majority)
        after = mai(stage2_choices[agent_id], majority)
        deltas.append(after - before)
    return mean(deltas)


def switch_rate(
    stage1_choices: Dict[str, str],
    stage2_choices: Dict[str, str],
    majority: str,
) -> float:
    agents = minority_agents(stage1_choices, majority)
    switches = [stage2_choices[agent_id] == majority for agent_id in agents]
    return mean(1.0 if switched else 0.0 for switched in switches)


def mean_minority_stage_mai(
    stage_choices: Dict[str, str],
    minority: List[str],
    majority: str,
) -> float:
    return mean(mai(stage_choices[agent_id], majority) for agent_id in minority)
