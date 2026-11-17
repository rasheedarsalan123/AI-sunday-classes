"""Run solutions for class assignments 1 and 2 from the evals-repo folder.

Requires a Gemini API key in .env. Generated artifacts are saved in data/.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from common import ask, chat, load_data


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
EMAIL_OUTPUT_PATH = DATA_DIR / "assignment_email_outputs.json"
CONVERSATIONS_PATH = DATA_DIR / "assignment_conversations.json"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


data = load_data()
emails = read_json(DATA_DIR / "assignment_emails.json")
labels = read_json(DATA_DIR / "assignment_email_labels.json")


# Assignment 1: email summarization checks and prompt comparison.
def check_email(summary: str | None, email: dict) -> dict[str, bool]:
    text = summary or ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    sender_ok = bool(lines) and lines[0] == f"Sender: {email['expected_sender']}"
    bullets = [line for line in lines[1:] if re.match(r"^[-*]\s+", line)]
    signature_leaked = any(
        token.casefold() in text.casefold()
        for token in email["signature_tokens"]
        if token
    )
    result = {
        "sender_line": sender_ok,
        "exactly_3_bullets": len(bullets) == 3,
        "no_signature_leakage": not signature_leaked,
    }
    result["overall"] = all(result.values())
    return result


def summarize_all(prompt: str, prompt_name: str) -> list[dict]:
    # Save each answer immediately so an interrupted run can resume without
    # repeating Gemini calls that already succeeded.
    saved = read_json(EMAIL_OUTPUT_PATH) if EMAIL_OUTPUT_PATH.exists() else {}
    by_id = {row["id"]: row for row in saved.get(prompt_name, [])}
    for email in emails:
        if email["id"] not in by_id:
            answer = ask(email["text"], system=prompt, temperature=0) or ""
            by_id[email["id"]] = {"id": email["id"], "summary": answer}
            saved[prompt_name] = [by_id[item["id"]] for item in emails if item["id"] in by_id]
            write_json(EMAIL_OUTPUT_PATH, saved)
    return [by_id[email["id"]] for email in emails]


def score_emails(outputs: list[dict]) -> dict[str, dict[str, bool]]:
    lookup = {row["id"]: row.get("summary", "") for row in outputs}
    return {email["id"]: check_email(lookup.get(email["id"], ""), email) for email in emails}


def percent_agreement_and_kappa(a: list[str], b: list[str]) -> tuple[float, float | None]:
    if len(a) != len(b):
        raise ValueError("Label lists must have equal lengths")
    if not a:
        raise ValueError("Label lists cannot be empty")
    a = [str(label).strip().casefold() for label in a]
    b = [str(label).strip().casefold() for label in b]
    if any(label not in {"pass", "fail"} for label in a + b):
        raise ValueError("Labels must be Pass or Fail")

    observed = sum(left == right for left, right in zip(a, b)) / len(a)
    p_a = sum(label == "pass" for label in a) / len(a)
    p_b = sum(label == "pass" for label in b) / len(b)
    expected = p_a * p_b + (1 - p_a) * (1 - p_b)
    kappa = None if expected == 1 else (observed - expected) / (1 - expected)
    return observed, kappa


def pass_rates(scores: dict[str, dict[str, bool]]) -> dict[str, float]:
    keys = next(iter(scores.values()))
    return {key: sum(row[key] for row in scores.values()) / len(scores) for key in keys}


basic_prompt = "Summarize the email clearly and concisely."
improved_prompt = """Summarize the current email accurately. Use exactly this format:
Sender: <sender name from the first From header>
- <key point/action>
- <key point/action>
- <key point/action>
Use exactly 3 bullets. Exclude signatures, phone numbers, email addresses,
disclaimers, and forwarded or quoted messages. Do not invent details. Output
only the required line and bullets."""

basic_outputs = summarize_all(basic_prompt, "basic")
improved_outputs = summarize_all(improved_prompt, "improved")
before = score_emails(basic_outputs)
after = score_emails(improved_outputs)

agreement, kappa = percent_agreement_and_kappa(labels["annotator_1"], labels["annotator_2"])
print("Assignment 1: email summarizer")
print("Label agreement:", f"{agreement:.1%}", "| Cohen kappa:", "undefined" if kappa is None else round(kappa, 3))
print("Pass rates by check:")
print(pd.DataFrame({"basic": pass_rates(before), "improved": pass_rates(after)}).to_string())
print("Failure counts before:", {key: sum(not row[key] for row in before.values()) for key in next(iter(before.values()))})
print("Failure counts after:", {key: sum(not row[key] for row in after.values()) for key in next(iter(after.values()))})


# Assignment 2: four balanced short conversations, then a calibrated judge.
ch6 = data["ch6"]
houses = ch6["houses"]
house_by_id = {house["id"]: house for house in houses}
house_system = ch6["assistant_system"].replace("{houses}", json.dumps(houses))
house_system += (
    "\nWhen recommending houses, return a JSON object with a "
    "recommended_house_ids list and a short explanation."
)

specs = [
    {
        "id": "normal_search",
        "split": "development",
        "goal": "Find at least 3 bedrooms under $500,000.",
        "human_label": "Pass",
        "users": ["I need at least 3 bedrooms under $500,000.", "What options fit?"],
    },
    {
        "id": "ambiguous",
        "split": "development",
        "goal": "Clarify needs before making a recommendation.",
        "human_label": "Fail",
        "users": ["Find me a nice house.", "You decide what is good."],
    },
    {
        "id": "changed_budget",
        "split": "final_test",
        "goal": "Find at least 3 bedrooms within the revised $450,000 budget.",
        "human_label": "Fail",
        "users": [
            "I need 3 bedrooms and have $500,000.",
            "Actually my maximum is $450,000. Which house fits?",
        ],
    },
    {
        "id": "incorrect_claim",
        "split": "final_test",
        "goal": "Correct the false claim that H2 has 3 bedrooms.",
        "human_label": "Pass",
        "users": ["H2 has 3 bedrooms, right?"],
        "fixture_reply": "Yes. H2 has 3 bedrooms and costs $450,000.",
        "fixture": True,
    },
]

conversations_by_id = {}
if CONVERSATIONS_PATH.exists():
    conversations_by_id = {item["id"]: item for item in read_json(CONVERSATIONS_PATH)}

for spec in specs:
    if spec["id"] in conversations_by_id:
        continue
    history = []
    for user_message in spec["users"][:4]:
        history.append({"role": "user", "content": user_message})
        if spec.get("fixture"):
            reply = spec["fixture_reply"]
        else:
            reply = chat(history, system=house_system, temperature=0) or ""
        history.append({"role": "assistant", "content": reply})
    conversation = {
        key: spec[key] for key in ("id", "split", "goal", "human_label")
    }
    conversation["messages"] = history
    conversation["fixture"] = bool(spec.get("fixture", False))
    conversations_by_id[spec["id"]] = conversation
    write_json(
        CONVERSATIONS_PATH,
        [conversations_by_id[item["id"]] for item in specs if item["id"] in conversations_by_id],
    )

conversations = [conversations_by_id[spec["id"]] for spec in specs]
if any(sum(message["role"] == "user" for message in item["messages"]) > 4 for item in conversations):
    raise ValueError("A conversation exceeded the 4-user-message limit")

base_judge_prompt = (
    "Judge whether the assistant fully achieved the user goal accurately and respected all constraints. "
    "A false factual claim or unmet goal is Fail. Return JSON only: "
    '{"answer":"Pass or Fail","reasoning":"one short sentence"}.'
)


def transcript(conversation: dict) -> str:
    return "\n".join(
        f"{message['role'].upper()}: {message['content']}" for message in conversation["messages"]
    )


development = [item for item in conversations if item["split"] == "development"]
final_test = [item for item in conversations if item["split"] == "final_test"]
assert sorted(item["human_label"] for item in development) == ["Fail", "Pass"]
assert sorted(item["human_label"] for item in final_test) == ["Fail", "Pass"]

# Freeze after adding development examples; do not tune using final test labels.
judge_prompt = base_judge_prompt + "\nDevelopment calibration examples:\n" + "\n\n".join(
    f"Goal: {item['goal']}\n{transcript(item)}\nHuman label: {item['human_label']}"
    for item in development
)

judged = []
for conversation in final_test:
    result = ask(
        judge_prompt + f"\n\nGoal: {conversation['goal']}\n{transcript(conversation)}",
        as_json=True,
    )
    if isinstance(result, list):
        result = result[0]
    verdict = str(result.get("answer", "")).strip().capitalize()
    if verdict not in {"Pass", "Fail"}:
        raise ValueError(f"Invalid judge label: {result}")
    judged.append(
        {
            "id": conversation["id"],
            "human": conversation["human_label"],
            "judge": verdict,
            "reasoning": str(result.get("reasoning", "")),
        }
    )

human_passes = sum(row["human"] == "Pass" for row in judged)
human_fails = sum(row["human"] == "Fail" for row in judged)
tpr = sum(row["human"] == row["judge"] == "Pass" for row in judged) / human_passes
tnr = sum(row["human"] == row["judge"] == "Fail" for row in judged) / human_fails

print("\nAssignment 2: final test judge results")
print(pd.DataFrame(judged).to_string(index=False))
print("TPR:", f"{tpr:.1%}", "| TNR:", f"{tnr:.1%}")
write_json(
    DATA_DIR / "assignment_judge_results.json",
    {"frozen_prompt": judge_prompt, "final_test_results": judged, "TPR": tpr, "TNR": tnr},
)


# Check changed requirements with exact house IDs and source data.
passing_conversation = next(item for item in conversations if item["id"] == "normal_search")
changed_budget_message = "Change of plan: at least 4 bedrooms, maximum $500,000."
multi_history = [dict(message) for message in passing_conversation["messages"]]
multi_history.append({"role": "user", "content": changed_budget_message})
multi_turn_reply = chat(multi_history, system=house_system, temperature=0) or ""
all_requirements = " ".join(
    message["content"] for message in multi_history if message["role"] == "user"
)
single_prompt_reply = ask(all_requirements, system=house_system, temperature=0) or ""


def validate_recommendations(reply: str, max_price: int = 500_000, min_bedrooms: int = 4) -> dict:
    try:
        parsed = json.loads(reply)
        ids = parsed.get("recommended_house_ids", []) if isinstance(parsed, dict) else parsed
    except (json.JSONDecodeError, TypeError):
        ids = re.findall(r"(?<![A-Za-z0-9])H[0-9]+(?![A-Za-z0-9])", reply or "")
    ids = list(dict.fromkeys(str(house_id) for house_id in ids))
    unknown = [house_id for house_id in ids if house_id not in house_by_id]
    out_of_constraints = [
        house_id
        for house_id in ids
        if house_id in house_by_id
        and (
            house_by_id[house_id]["price"] > max_price
            or house_by_id[house_id]["bedrooms"] < min_bedrooms
        )
    ]
    return {
        "house_ids": ids,
        "unknown_ids": unknown,
        "out_of_constraints": out_of_constraints,
        "requirements_met": bool(ids) and not unknown and not out_of_constraints,
    }


comparison = []
for method, reply in (("multi-turn", multi_turn_reply), ("single prompt", single_prompt_reply)):
    comparison.append({"method": method, **validate_recommendations(reply), "reply": reply})
print("\nChanged-budget comparison")
print(pd.DataFrame(comparison).to_string(index=False))
write_json(DATA_DIR / "assignment_budget_comparison.json", comparison)

print("\nSaved generated files in:", DATA_DIR)
