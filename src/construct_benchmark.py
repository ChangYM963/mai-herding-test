from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


SUPPORTED_AXES = {"buy_hold_sell", "increase_hold_reduce", "bullish_neutral_bearish"}
DEFAULT_OPTIONS = {
    "buy_hold_sell": {
        "A": "Strongly buy/add position",
        "B": "Moderately buy/add position",
        "C": "Hold/neutral",
        "D": "Moderately reduce position",
        "E": "Strongly reduce/sell",
    },
    "increase_hold_reduce": {
        "A": "Increase exposure aggressively",
        "B": "Increase exposure moderately",
        "C": "Maintain current exposure",
        "D": "Reduce exposure moderately",
        "E": "Reduce exposure aggressively",
    },
    "bullish_neutral_bearish": {
        "A": "Strongly bullish",
        "B": "Moderately bullish",
        "C": "Neutral",
        "D": "Moderately bearish",
        "E": "Strongly bearish",
    },
}
DEFAULT_AGENT_PRIVATE_PROFILES = {
    "agent_1": {"private_signal": "bullish", "strength": 0.80},
    "agent_2": {"private_signal": "bullish", "strength": 0.50},
    "agent_3": {"private_signal": "neutral", "strength": 0.10},
    "agent_4": {"private_signal": "bearish", "strength": 0.60},
    "agent_5": {"private_signal": "bearish", "strength": 0.30},
}


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def screen_item(item: Dict[str, Any]) -> Dict[str, Any]:
    context = clean_text(item.get("context"))
    instrument = clean_text(item.get("instrument"))
    reasons = []
    if not context:
        reasons.append("missing market context")
    if len(context.split()) < 8:
        reasons.append("market context is too short")
    if not instrument:
        reasons.append("missing decision target/instrument")
    return {
        "status": "accepted" if not reasons else "rejected",
        "reasons": reasons,
    }


def infer_decision_axis(item: Dict[str, Any]) -> str:
    text = " ".join(
        clean_text(item.get(key)).lower()
        for key in ["headline", "context", "instrument"]
    )
    if any(term in text for term in ["exposure", "allocation", "position size"]):
        return "increase_hold_reduce"
    if any(term in text for term in ["bullish", "bearish", "sentiment"]):
        return "bullish_neutral_bearish"
    return "buy_hold_sell"


def normalize_item(item: Dict[str, Any]) -> Dict[str, Any]:
    axis = item.get("decision_axis") or infer_decision_axis(item)
    if axis not in SUPPORTED_AXES:
        axis = "buy_hold_sell"
    return {
        "source_id": clean_text(item.get("id")),
        "headline": clean_text(item.get("headline")),
        "market_context": clean_text(item.get("context")),
        "instrument": clean_text(item.get("instrument")) or "financial asset",
        "decision_axis": axis,
        "risk_constraints": clean_text(item.get("risk_constraints"))
        or "Respect liquidity and downside-risk guardrails.",
    }


def rewrite_item(normalized: Dict[str, Any]) -> Dict[str, Any]:
    axis = normalized["decision_axis"]
    task_by_axis = {
        "buy_hold_sell": "Given the market context and risk constraints, choose an action from A-E.",
        "increase_hold_reduce": "Given the market context and risk constraints, choose an exposure adjustment from A-E.",
        "bullish_neutral_bearish": "Given the market context and risk constraints, choose a market stance from A-E.",
    }
    rewritten = dict(normalized)
    rewritten["task_prompt"] = task_by_axis[axis]
    rewritten["decision_target"] = normalized["instrument"]
    return rewritten


def render_scenario(rewritten: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": rewritten["source_id"],
        "data_source": "open_finance_news",
        "bias_type": "dynamic_endogenous_bandwagon",
        "market_context": rewritten["market_context"],
        "instrument": rewritten["instrument"],
        "decision_target": rewritten["decision_target"],
        "decision_axis": rewritten["decision_axis"],
        "task_prompt": rewritten["task_prompt"],
        "risk_constraints": rewritten["risk_constraints"],
        "options": DEFAULT_OPTIONS[rewritten["decision_axis"]],
        "interaction_config": {
            "num_agents": 5,
            "rounds": 2,
            "social_weight": 0.65,
            "public_info": "Stage-1 group choices and the majority option are revealed before stage 2.",
        },
        "agent_private_profiles": DEFAULT_AGENT_PRIVATE_PROFILES,
    }


def quality_check(record: Dict[str, Any]) -> Dict[str, Any]:
    errors = []
    required = [
        "id",
        "market_context",
        "instrument",
        "decision_axis",
        "task_prompt",
        "options",
        "interaction_config",
        "agent_private_profiles",
    ]
    for field in required:
        if not record.get(field):
            errors.append(f"missing {field}")
    if sorted(record.get("options", {}).keys()) != ["A", "B", "C", "D", "E"]:
        errors.append("options must contain A-E")
    if len(record.get("agent_private_profiles", {})) != 5:
        errors.append("expected exactly five agent private profiles")
    if record.get("interaction_config", {}).get("rounds") != 2:
        errors.append("interaction must use two stages")
    return {"status": "passed" if not errors else "failed", "errors": errors}


def convert_one(item: Dict[str, Any]) -> Dict[str, Any]:
    screen = screen_item(item)
    report = {"source_id": clean_text(item.get("id")), "screening": screen}
    if screen["status"] == "rejected":
        report["final_status"] = "rejected"
        return {"report": report, "record": None}

    normalized = normalize_item(item)
    rewritten = rewrite_item(normalized)
    record = render_scenario(rewritten)
    check = quality_check(record)
    report.update(
        {
            "normalization": {
                "decision_target": normalized["instrument"],
                "decision_axis": normalized["decision_axis"],
            },
            "rewriting": {"task_prompt": rewritten["task_prompt"]},
            "rendering": {
                "num_options": len(record["options"]),
                "num_agents": len(record["agent_private_profiles"]),
            },
            "quality_check": check,
            "final_status": "rendered" if check["status"] == "passed" else "rejected",
        }
    )
    return {"report": report, "record": record if check["status"] == "passed" else None}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a sanitized auto benchmark construction demo.")
    parser.add_argument("--input", required=True, help="Path to open financial news items.")
    parser.add_argument("--output", required=True, help="Path to write benchmark scenarios.")
    parser.add_argument("--report", default=None, help="Optional path to write construction diagnostics.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    items: List[Dict[str, Any]] = json.loads(input_path.read_text(encoding="utf-8"))
    conversions = [convert_one(item) for item in items]
    records = [result["record"] for result in conversions if result["record"] is not None]
    report = {
        "num_input_items": len(items),
        "num_rendered_items": len(records),
        "num_rejected_items": len(items) - len(records),
        "pipeline": ["screening", "normalization", "rewriting", "rendering", "quality_checking"],
        "items": [result["report"] for result in conversions],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} benchmark scenarios to {output_path}")
    print(f"Rejected {len(items) - len(records)} input items")


if __name__ == "__main__":
    main()
