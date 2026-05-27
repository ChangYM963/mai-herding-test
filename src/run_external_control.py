from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from metrics import (
    OPTION_ORDER,
    agent_mai_table,
    majority_option,
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


VALID_ROUTES = {"prompt-only", "posthoc-only"}
CONTROL_PROTOCOL_SPEC = {
    "protocol_version": "CONTROL_PROTOCOL_V1_PUBLIC_DEMO",
    "task": "endogenous_bandwagon_control",
    "control_target": "delta_mai_to_zero",
    "control_object": "probability_shift_not_vote_flip",
    "output_contract": "respond with exactly one option letter (A/B/C/D/E)",
}


def clamp_control_level(control_level: float) -> float:
    return round(max(0.0, min(1.0, float(control_level))) * 10.0) / 10.0


def build_control_protocol_block(majority: str, stage1_choice: str, control_level: float) -> Dict:
    return {
        **CONTROL_PROTOCOL_SPEC,
        "control_level": clamp_control_level(control_level),
        "control_route": "prompt-only",
        "majority_option": majority,
        "stage1_choice": stage1_choice,
    }


def prompt_route_follow_probability(scenario: Dict, control_level: float) -> float:
    baseline = float(scenario.get("interaction_config", {}).get("social_weight", 0.65))
    return baseline * (1.0 - 0.60 * clamp_control_level(control_level))


def shift_one_step_toward(source: str, target: str) -> str:
    source_idx = OPTION_ORDER.index(source)
    target_idx = OPTION_ORDER.index(target)
    if source_idx == target_idx:
        return source
    if source_idx < target_idx:
        return OPTION_ORDER[source_idx + 1]
    return OPTION_ORDER[source_idx - 1]


def posthoc_shrink_choice(stage1_choice: str, raw_stage2_choice: str, control_level: float) -> str:
    final_choice = raw_stage2_choice
    steps = int(round(clamp_control_level(control_level) * 2))
    for _ in range(steps):
        final_choice = shift_one_step_toward(final_choice, stage1_choice)
    return final_choice


def run_prompt_only_scenario(scenario: Dict, num_agents: int, control_level: float) -> Dict:
    agent_ids = agent_ids_for_scenario(scenario, num_agents)
    stage1 = {
        agent_id: mock_stage1_choice(scenario["id"], agent_id, profile_for_agent(scenario, agent_id))
        for agent_id in agent_ids
    }
    majority = majority_option(stage1)
    minority = minority_agents(stage1, majority)
    follow_probability = prompt_route_follow_probability(scenario, control_level)
    stage2 = {
        agent_id: mock_stage2_choice(
            scenario["id"],
            agent_id,
            stage1[agent_id],
            majority,
            follow_probability,
        )
        for agent_id in agent_ids
    }
    protocol_blocks = {
        agent_id: build_control_protocol_block(majority, stage1[agent_id], control_level)
        for agent_id in agent_ids
    }
    return build_controlled_result(
        scenario=scenario,
        route="prompt-only",
        stage1=stage1,
        stage2_final=stage2,
        majority=majority,
        minority=minority,
        diagnostics={
            "prompt_control_protocol": protocol_blocks,
            "raw_stage2_choices": stage2,
            "final_stage2_choices": stage2,
            "note": "Prompt-only adds the structured control protocol before stage-2 generation; no posthoc shrinkage is applied.",
        },
    )


def run_posthoc_only_scenario(scenario: Dict, num_agents: int, control_level: float) -> Dict:
    agent_ids = agent_ids_for_scenario(scenario, num_agents)
    stage1 = {
        agent_id: mock_stage1_choice(scenario["id"], agent_id, profile_for_agent(scenario, agent_id))
        for agent_id in agent_ids
    }
    majority = majority_option(stage1)
    minority = minority_agents(stage1, majority)
    baseline_follow_probability = float(scenario.get("interaction_config", {}).get("social_weight", 0.65))
    raw_stage2 = {
        agent_id: mock_stage2_choice(
            scenario["id"],
            agent_id,
            stage1[agent_id],
            majority,
            baseline_follow_probability,
        )
        for agent_id in agent_ids
    }
    final_stage2 = {
        agent_id: posthoc_shrink_choice(stage1[agent_id], raw_stage2[agent_id], control_level)
        for agent_id in agent_ids
    }
    return build_controlled_result(
        scenario=scenario,
        route="posthoc-only",
        stage1=stage1,
        stage2_final=final_stage2,
        majority=majority,
        minority=minority,
        diagnostics={
            "raw_stage2_choices": raw_stage2,
            "final_stage2_choices": final_stage2,
            "posthoc_rule": "choice-level public demo of P2_ctrl = P1 + (1-c) * (P2_raw - P1)",
            "note": "Posthoc-only leaves the stage-2 prompt unchanged and applies numeric shrinkage after raw stage-2 output.",
        },
    )


def build_controlled_result(
    scenario: Dict,
    route: str,
    stage1: Dict[str, str],
    stage2_final: Dict[str, str],
    majority: str,
    minority: List[str],
    diagnostics: Dict,
) -> Dict:
    stage1_minority_mai = mean_minority_stage_mai(stage1, minority, majority)
    stage2_minority_mai = mean_minority_stage_mai(stage2_final, minority, majority)
    return {
        "scenario_id": scenario["id"],
        "route": route,
        "stage1_independent_choices": {
            "choices": stage1,
            "option_counts": option_counts(stage1),
        },
        "endogenous_majority": {
            "majority_option": majority,
            "formation_rule": "argmax over stage-1 option counts",
        },
        "fixed_minority_group": {
            "agent_ids": minority,
        },
        "external_control": diagnostics,
        "stage2_controlled_choices": {
            "choices": stage2_final,
            "option_counts": option_counts(stage2_final),
        },
        "agent_level_MAI": agent_mai_table(stage1, stage2_final, majority, minority),
        "scenario_summary": {
            "minority_MAI_stage1": stage1_minority_mai,
            "minority_MAI_stage2": stage2_minority_mai,
            "minority_delta_MAI": minority_delta_mai(stage1, stage2_final, majority),
            "minority_switch_rate_to_majority": switch_rate(stage1, stage2_final, majority),
        },
    }


def run_controlled_scenario(scenario: Dict, num_agents: int, route: str, control_level: float) -> Dict:
    if route == "prompt-only":
        return run_prompt_only_scenario(scenario, num_agents, control_level)
    if route == "posthoc-only":
        return run_posthoc_only_scenario(scenario, num_agents, control_level)
    raise ValueError(f"Unsupported route: {route}. Expected one of {sorted(VALID_ROUTES)}.")


def pairwise_effects(baseline_rows: List[Dict], controlled_rows: List[Dict]) -> List[Dict]:
    effects = []
    for baseline, controlled in zip(baseline_rows, controlled_rows):
        base_summary = baseline["scenario_summary"]
        controlled_summary = controlled["scenario_summary"]
        effects.append(
            {
                "scenario_id": baseline["scenario_id"],
                "baseline_minority_delta_MAI": base_summary["minority_delta_MAI"],
                "controlled_minority_delta_MAI": controlled_summary["minority_delta_MAI"],
                "delta_reduction": base_summary["minority_delta_MAI"]
                - controlled_summary["minority_delta_MAI"],
                "baseline_switch_rate": base_summary["minority_switch_rate_to_majority"],
                "controlled_switch_rate": controlled_summary["minority_switch_rate_to_majority"],
            }
        )
    return effects


def summarize_control(baseline_rows: List[Dict], controlled_rows: List[Dict]) -> Dict:
    baseline_summary = summarize(baseline_rows)
    controlled_summary = summarize(controlled_rows)
    return {
        "baseline": baseline_summary,
        "external_control": controlled_summary,
        "mean_minority_delta_MAI_reduction": baseline_summary["mean_minority_delta_MAI"]
        - controlled_summary["mean_minority_delta_MAI"],
        "mean_switch_rate_reduction": baseline_summary["mean_minority_switch_rate_to_majority"]
        - controlled_summary["mean_minority_switch_rate_to_majority"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a sanitized external-control MAI demo.")
    parser.add_argument("--data", required=True, help="Path to benchmark scenarios.")
    parser.add_argument("--num-agents", type=int, default=5)
    parser.add_argument("--route", choices=sorted(VALID_ROUTES), default="posthoc-only")
    parser.add_argument("--control-level", type=float, default=0.5)
    parser.add_argument("--output", default="results/external_control_demo.json")
    args = parser.parse_args()

    control_level = clamp_control_level(args.control_level)
    scenarios = json.loads(Path(args.data).read_text(encoding="utf-8"))
    baseline_rows = [run_scenario(scenario, args.num_agents) for scenario in scenarios]
    controlled_rows = [
        run_controlled_scenario(scenario, args.num_agents, args.route, control_level)
        for scenario in scenarios
    ]
    payload = {
        "protocol": {
            "metric": "MAI",
            "control_type": "external control",
            "route": args.route,
            "comparison": "baseline two-stage validation vs controlled stage-2 validation",
            "reported_effect": "reduction in minority_delta_MAI under the selected route",
        },
        "control_config": {
            "control_level": control_level,
            "route_definitions": {
                "prompt-only": "structured, quantized control protocol inserted into stage-2 prompt; no posthoc shrinkage",
                "posthoc-only": "unchanged stage-2 prompt followed by posthoc shrinkage toward stage-1 probabilities",
            },
            "note": "This public demo uses choices rather than full probability vectors, but preserves the route separation.",
        },
        "summary": summarize_control(baseline_rows, controlled_rows),
        "scenario_effects": pairwise_effects(baseline_rows, controlled_rows),
        "baseline_scenarios": baseline_rows,
        "controlled_scenarios": controlled_rows,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"Wrote external-control details to {output_path}")


if __name__ == "__main__":
    main()

