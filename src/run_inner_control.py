from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List

from metrics import (
    OPTION_ORDER,
    agent_mai_table,
    majority_option,
    mean,
    mean_minority_stage_mai,
    minority_agents,
    minority_delta_mai,
    option_counts,
    switch_rate,
)
from run_validation import (
    agent_ids_for_scenario,
    mock_stage1_choice,
    mock_stage2_choice,
    profile_for_agent,
    run_scenario,
    summarize,
)


VECTOR_DIM = 8


def stable_unit(seed: str) -> float:
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return value * 2.0 - 1.0


def normalize(vector: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm <= 1e-12:
        return [0.0 for _ in vector]
    return [v / norm for v in vector]


def add_vectors(a: List[float], b: List[float]) -> List[float]:
    return [x + y for x, y in zip(a, b)]


def sub_vectors(a: List[float], b: List[float]) -> List[float]:
    return [x - y for x, y in zip(a, b)]


def scale_vector(vector: List[float], scale: float) -> List[float]:
    return [scale * v for v in vector]


def dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def option_latent(option: str) -> List[float]:
    option_idx = OPTION_ORDER.index(option)
    polarity = 1.0 - option_idx / 2.0
    return normalize([
        polarity,
        0.5 * polarity,
        stable_unit(f"{option}:0"),
        stable_unit(f"{option}:1"),
        stable_unit(f"{option}:2"),
        stable_unit(f"{option}:3"),
        stable_unit(f"{option}:4"),
        stable_unit(f"{option}:5"),
    ])


def hidden_proxy(scenario_id: str, agent_id: str, stage: str, choice: str) -> List[float]:
    base = option_latent(choice)
    noise = [0.08 * stable_unit(f"{scenario_id}:{agent_id}:{stage}:{idx}") for idx in range(VECTOR_DIM)]
    return normalize(add_vectors(base, noise))


def choice_from_hidden(hidden: List[float]) -> str:
    scores = {option: dot(hidden, option_latent(option)) for option in OPTION_ORDER}
    return max(OPTION_ORDER, key=lambda option: (scores[option], -OPTION_ORDER.index(option)))


def mean_vector(vectors: Iterable[List[float]]) -> List[float]:
    vectors = list(vectors)
    if not vectors:
        return [0.0] * VECTOR_DIM
    return [sum(vector[idx] for vector in vectors) / len(vectors) for idx in range(VECTOR_DIM)]


def build_herd_direction(baseline_rows: List[Dict]) -> List[float]:
    diffs = []
    for row in baseline_rows:
        for agent_id in row["fixed_minority_group"]["agent_ids"]:
            hidden = row["internal_states"][agent_id]
            diffs.append(sub_vectors(hidden["stage2_raw_hidden"], hidden["stage1_hidden"]))
    return normalize(mean_vector(diffs))


def risk_score_from_gate(gate_raw: float, gate_scale: float) -> float:
    return max(0.0, min(1.0, gate_raw * gate_scale))


def run_baseline_with_hidden(scenario: Dict, num_agents: int) -> Dict:
    row = run_scenario(scenario, num_agents)
    stage1 = row["stage1_independent_choices"]["choices"]
    stage2 = row["stage2_social_exposure"]["choices"]
    internal_states = {}
    for agent_id in stage1:
        h1 = hidden_proxy(scenario["id"], agent_id, "stage1", stage1[agent_id])
        h2 = hidden_proxy(scenario["id"], agent_id, "stage2_raw", stage2[agent_id])
        internal_states[agent_id] = {
            "stage1_hidden": h1,
            "stage2_raw_hidden": h2,
        }
    row["internal_states"] = internal_states
    return row


def run_inner_control_scenario(
    scenario: Dict,
    baseline_row: Dict,
    herd_direction: List[float],
    alpha: float,
    gate_threshold: float,
    gate_scale: float,
) -> Dict:
    stage1 = baseline_row["stage1_independent_choices"]["choices"]
    raw_stage2 = baseline_row["stage2_social_exposure"]["choices"]
    majority = baseline_row["endogenous_majority"]["majority_option"]
    minority = minority_agents(stage1, majority)
    final_stage2 = {}
    internal_meta = {}

    for agent_id, stage1_choice in stage1.items():
        h1 = hidden_proxy(scenario["id"], agent_id, "stage1", stage1_choice)
        h2_raw = hidden_proxy(scenario["id"], agent_id, "stage2_raw", raw_stage2[agent_id])
        is_minority = agent_id in minority
        gate_raw = max(
            0.0,
            dot(h2_raw, option_latent(majority)) - dot(h1, option_latent(majority)),
        )
        steering_applied = is_minority and gate_raw > gate_threshold and alpha > 0
        if steering_applied:
            # Public hidden proxies are low-dimensional, so we use a fixed demo scale
            # to make alpha values comparable to the full implementation's grid.
            strength = 4.0 * alpha * risk_score_from_gate(gate_raw, gate_scale)
            h2_final = normalize(sub_vectors(h2_raw, scale_vector(herd_direction, strength)))
        else:
            strength = 0.0
            h2_final = h2_raw

        final_stage2[agent_id] = choice_from_hidden(h2_final)
        internal_meta[agent_id] = {
            "is_stage1_minority": is_minority,
            "gate_raw": gate_raw,
            "gate_threshold": gate_threshold,
            "alpha": alpha if steering_applied else 0.0,
            "control_strength": strength,
            "minority_control_applied": steering_applied,
        }

    stage1_minority_mai = mean_minority_stage_mai(stage1, minority, majority)
    stage2_minority_mai = mean_minority_stage_mai(final_stage2, minority, majority)
    return {
        "scenario_id": scenario["id"],
        "control_type": "inner representation control",
        "stage1_independent_choices": baseline_row["stage1_independent_choices"],
        "endogenous_majority": baseline_row["endogenous_majority"],
        "fixed_minority_group": baseline_row["fixed_minority_group"],
        "inner_control": {
            "target": "stage-2 hidden representation of stage-1 minority agents",
            "control_mode": "conditional",
            "alpha": alpha,
            "gate_threshold": gate_threshold,
            "gate_scale": gate_scale,
            "steering_direction": "negative herd direction estimated from baseline minority hidden-state drift",
            "agent_internal_meta": internal_meta,
        },
        "stage2_raw_choices": raw_stage2,
        "stage2_controlled_choices": {
            "choices": final_stage2,
            "option_counts": option_counts(final_stage2),
        },
        "agent_level_MAI": agent_mai_table(stage1, final_stage2, majority, minority),
        "scenario_summary": {
            "minority_MAI_stage1": stage1_minority_mai,
            "minority_MAI_stage2": stage2_minority_mai,
            "minority_delta_MAI": minority_delta_mai(stage1, final_stage2, majority),
            "minority_switch_rate_to_majority": switch_rate(stage1, final_stage2, majority),
        },
    }


def pairwise_effects(baseline_rows: List[Dict], controlled_rows: List[Dict]) -> List[Dict]:
    effects = []
    for baseline, controlled in zip(baseline_rows, controlled_rows):
        b = baseline["scenario_summary"]
        c = controlled["scenario_summary"]
        effects.append(
            {
                "scenario_id": baseline["scenario_id"],
                "baseline_minority_delta_MAI": b["minority_delta_MAI"],
                "inner_control_minority_delta_MAI": c["minority_delta_MAI"],
                "delta_reduction": b["minority_delta_MAI"] - c["minority_delta_MAI"],
                "baseline_switch_rate": b["minority_switch_rate_to_majority"],
                "inner_control_switch_rate": c["minority_switch_rate_to_majority"],
            }
        )
    return effects


def summarize_inner_control(baseline_rows: List[Dict], controlled_rows: List[Dict]) -> Dict:
    baseline_summary = summarize(baseline_rows)
    controlled_summary = summarize(controlled_rows)
    return {
        "baseline": baseline_summary,
        "inner_control": controlled_summary,
        "mean_minority_delta_MAI_reduction": baseline_summary["mean_minority_delta_MAI"]
        - controlled_summary["mean_minority_delta_MAI"],
        "mean_switch_rate_reduction": baseline_summary["mean_minority_switch_rate_to_majority"]
        - controlled_summary["mean_minority_switch_rate_to_majority"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a sanitized inner-control MAI demo.")
    parser.add_argument("--data", required=True, help="Path to benchmark scenarios.")
    parser.add_argument("--num-agents", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--gate-threshold", type=float, default=0.0)
    parser.add_argument("--gate-scale", type=float, default=1.0)
    parser.add_argument("--output", default="results/inner_control_demo.json")
    args = parser.parse_args()

    scenarios = json.loads(Path(args.data).read_text(encoding="utf-8"))
    baseline_rows = [run_baseline_with_hidden(scenario, args.num_agents) for scenario in scenarios]
    herd_direction = build_herd_direction(baseline_rows)
    controlled_rows = [
        run_inner_control_scenario(
            scenario=scenario,
            baseline_row=baseline,
            herd_direction=herd_direction,
            alpha=max(0.0, float(args.alpha)),
            gate_threshold=max(0.0, float(args.gate_threshold)),
            gate_scale=max(0.0, float(args.gate_scale)),
        )
        for scenario, baseline in zip(scenarios, baseline_rows)
    ]
    payload = {
        "protocol": {
            "metric": "MAI",
            "control_type": "inner representation control",
            "public_demo_note": "This demo uses deterministic hidden-state proxies instead of real model activations.",
            "full_method_mapping": [
                "collect stage-1 and raw stage-2 hidden states",
                "estimate herd direction from minority hidden-state drift",
                "apply conditional steering at stage 2 for stage-1 minority agents",
                "compare baseline vs controlled minority_delta_MAI",
            ],
        },
        "control_config": {
            "control_mode": "conditional",
            "alpha": max(0.0, float(args.alpha)),
            "gate_threshold": max(0.0, float(args.gate_threshold)),
            "gate_scale": max(0.0, float(args.gate_scale)),
        },
        "summary": summarize_inner_control(baseline_rows, controlled_rows),
        "scenario_effects": pairwise_effects(baseline_rows, controlled_rows),
        "baseline_scenarios": baseline_rows,
        "controlled_scenarios": controlled_rows,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"Wrote inner-control details to {output_path}")


if __name__ == "__main__":
    main()
