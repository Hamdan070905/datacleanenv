"""
inference.py — Baseline inference script for DataCleanEnv.
Uses OpenAI client pointed at HuggingFace Inference API (free).
Output format strictly follows OpenEnv spec:
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

import os
import json
import requests
from openai import OpenAI

# ── Environment variables ────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")
ENV_URL      = os.getenv("ENV_URL",      "http://localhost:7860")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# ── OpenAI-compatible client pointed at HuggingFace ──────────────────────────
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

# ── Helpers ──────────────────────────────────────────────────────────────────

def env_reset(task_id: str) -> dict:
    r = requests.post(f"{ENV_URL}/reset", params={"task_id": task_id})
    r.raise_for_status()
    return r.json()

def env_step(action: dict) -> dict:
    r = requests.post(f"{ENV_URL}/step", json=action)
    r.raise_for_status()
    return r.json()

def env_state() -> dict:
    r = requests.get(f"{ENV_URL}/state")
    r.raise_for_status()
    return r.json()

def build_prompt(obs: dict) -> str:
    table_str = json.dumps(obs["current_table"], indent=2)
    return f"""You are a data cleaning agent. Fix data quality issues in this table.

Current table:
{table_str}

Issues remaining: {obs['issues_remaining']}
Last action result: {obs['last_action_result']}

Respond with EXACTLY one JSON object, no extra text:

To fill a missing value:
{{"action_type": "fill_missing", "row_index": <int>, "column": "<col>", "new_value": <value>}}

To fix a wrong data type:
{{"action_type": "fix_type", "row_index": <int>, "column": "<col>", "new_value": <value>}}

To remove a duplicate row:
{{"action_type": "remove_duplicate", "row_index": <int>}}

To fix an invalid value:
{{"action_type": "fix_value", "row_index": <int>, "column": "<col>", "new_value": <value>}}

When all issues are fixed:
{{"action_type": "done"}}

Rules:
- Fix ONE issue per response
- Look for: null values, duplicate rows, string instead of number, negative values
- When issues_remaining is 0, respond with {{"action_type": "done"}}
- Respond with ONLY the JSON. No explanation.
"""

def call_llm(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return '{"action_type": "done"}'

def parse_action(raw: str) -> dict:
    try:
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)
    except Exception:
        return {"action_type": "done"}

# ── Main episode runner ───────────────────────────────────────────────────────

def run_episode(task_id: str) -> dict:
    obs = env_reset(task_id)
    rewards = []
    MAX_STEPS = 25

    print(f"[START] task={task_id} env=datacleanenv model={MODEL_NAME}")

    for step_num in range(1, MAX_STEPS + 1):
        if obs.get("done"):
            break

        prompt = build_prompt(obs)
        raw_action = call_llm(prompt)
        action_dict = parse_action(raw_action)
        action_str = json.dumps(action_dict).replace(" ", "")

        try:
            result = env_step(action_dict)
            reward = result["reward"]["value"]
            done   = result["done"]
            obs    = result["observation"]
            error  = result["info"].get("error", "null") or "null"
        except Exception as e:
            reward = 0.0
            done   = False
            error  = str(e)

        rewards.append(reward)
        done_str  = "true" if done else "false"
        error_str = error if error and error != "null" else "null"

        print(
            f"[STEP] step={step_num} action={action_str} "
            f"reward={reward:.2f} done={done_str} error={error_str}"
        )

        if done:
            break

    success     = obs.get("issues_remaining", 1) == 0
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(f"[END] success={success_str} steps={len(rewards)} rewards={rewards_str}")

    return {"task_id": task_id, "success": success, "steps": len(rewards), "rewards": rewards}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tasks   = ["easy", "medium", "hard"]
    results = []

    for tid in tasks:
        res = run_episode(tid)
        results.append(res)
        print()

    print("=" * 50)
    print("BASELINE SCORES SUMMARY")
    print("=" * 50)
    for r in results:
        total_reward = sum(r["rewards"])
        print(f"Task: {r['task_id']:8s} | Steps: {r['steps']:3d} | "
              f"Success: {str(r['success']):5s} | Total Reward: {total_reward:.2f}")
