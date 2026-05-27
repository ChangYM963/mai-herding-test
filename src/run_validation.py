from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List

from metrics import (
    agent_mai_table,
    majority_option,
    mean_minority_stage_mai,
    minority_agents,
    minority_delta_mai,
    option_counts,
    switch_rate,
)


OPTIONS = ["A", "B", "C", "D", "E"]
PRIVATE_SIGNAL_PRIORS = {
    "bullish": ["A", "B", "B", "C"],
    "neutral": ["B", "C", "C", "D"],
    "bearish": ["C", "D", "D", "E"],
}


def stable_index(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def profile_for_agent(scenario: Dict, agent_id: str) -> Dict:
    return scenario.get("agent_private_profiles", {}).get(
        agent_id,
        {"private_signal": "neutral", "strength": 0.1},
    )


def mock_stage1_choice(scenario_id: str, agent_id: str, profile: Dict) -> str:
    signal = profile.get("private_signal", "neutral")
    candidates = PRIVATE_SIGNAL_PRIORS.get(signal, PRIVATE_SIGNAL_PRIORS["neutral"])
    return candidates[stable_index(scenario_id, agent_id, "stage1") % len(candidates)]


def move_one_step_toward_majority(choice: str, majority: str) -> str:
    choice_idx = OPTIONS.index(choice)
    majority_idx = OPTIONS.index(majority)
    if choice_idx == majority_idx:
        return choice
    if choice_idx < majority_idx:
        return OPTIONS[choice_idx + 1]
    return OPTIONS[choice_idx - 1]


def mock_stage2_choice(
    scenario_id: str,
    agent_id: str,
    stage1_choice: str,
    majority: str,
    social_weight: float,
) -> str:
    threshold = int(max(0.0, min(1.0, social_weight)) * 100)
    moves_toward_majority = stable_index(scenario_id, agent_id, "stage2") % 100 < threshold
    if moves_toward_majority:
        return move_one_step_toward_majority(stage1_choice, majority)
    return stage1_choice


def agent_ids_for_scenario(scenario: Dict, num_agents: int) -> List[str]:
    profiles = scenario.get("agent_private_profiles", {})
    if profiles:
        return list(profiles.keys())[:num_agents]
    return [f"agent_{idx}" for idx in range(1, num_agents + 1)]


def run_scenario(scenario: Dict, num_agents: int) -> Dict:
    agent_ids = agent_ids_for_scenario(scenario, num_agents)
    social_weight = scenario.get("interaction_config", {}).get("social_weight", 0.65)
    stage1 = {
        agent_id: mock_stage1_choice(scenario["id"], agent_id, profile_for_agent(scenario, agent_id))
        for agent_id in agent_ids
    }
    majority = majority_option(stage1)
    minority = minority_agents(stage1, majority)
    stage2 = {
        agent_id: mock_stage2_choice(
            scenario["id"],
            agent_id,
            stage1[agent_id],
            majority,
            social_weight,
        )
        for agent_id in agent_ids
    }
    stage1_minority_mai = mean_minority_stage_mai(stage1, minority, majority)
    stage2_minority_mai = mean_minority_stage_mai(stage2, minority, majority)
    return {
        "scenario_id": scenario["id"],
        "stage1_independent_choices": {
            "description": "Agents answer independently using only the market context and their private profiles.",
            "choices": stage1,
            "option_counts": option_counts(stage1),
        },
        "endogenous_majority": {
            "majority_option": majority,
            "formation_rule": "argmax over stage-1 option counts",
        },
        "fixed_minority_group": {
            "description": "Minority agents are fixed after stage 1 and reused for all MAI aggregation.",
            "agent_ids": minority,
        },
        "stage2_social_exposure": {
            "description": "Agents answer again after observing the stage-1 majority option.",
            "revealed_majority_option": majority,
            "social_weight": social_weight,
            "choices": stage2,
            "option_counts": option_counts(stage2),
        },
        "agent_level_MAI": agent_mai_table(stage1, stage2, majority, minority),
        "scenario_summary": {
            "minority_MAI_stage1": stage1_minority_mai,
            "minority_MAI_stage2": stage2_minority_mai,
            "minority_delta_MAI": minority_delta_mai(stage1, stage2, majority),
            "minority_switch_rate_to_majority": switch_rate(stage1, stage2, majority),
        },
    }


def summarize(rows: List[Dict]) -> Dict:
    if not rows:
        return {"num_scenarios": 0, "mean_minority_delta_MAI": 0.0, "mean_switch_rate": 0.0}
    return {
        "num_scenarios": len(rows),
        "mean_minority_delta_MAI": sum(row["scenario_summary"]["minority_delta_MAI"] for row in rows) / len(rows),
        "mean_minority_switch_rate_to_majority": sum(
            row["scenario_summary"]["minority_switch_rate_to_majority"] for row in rows
        ) / len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal two-stage bandwagon validation demo.")
    parser.add_argument("--data", required=True, help="Path to benchmark scenarios.")
    parser.add_argument("--num-agents", type=int, default=5)
    parser.add_argument("--output", default="results/demo_run.json")
    args = parser.parse_args()

    scenarios = json.loads(Path(args.data).read_text(encoding="utf-8"))
    rows = [run_scenario(scenario, args.num_agents) for scenario in scenarios]
    payload = {
        "protocol": {
            "metric": "MAI",
            "stage1": "independent choices",
            "majority": "endogenously formed from stage-1 choices",
            "minority_group": "agents not choosing the stage-1 majority option",
            "stage2": "choices after observing the stage-1 majority option",
            "aggregation": "mean delta_MAI over the fixed stage-1 minority group",
        },
        "summary": summarize(rows),
        "scenarios": rows,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"Wrote validation details to {output_path}")


if __name__ == "__main__":
    main()
