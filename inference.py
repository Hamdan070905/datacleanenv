"""
inference.py - Baseline inference script for DataCleanEnv.
Uses OpenAI client pointed at HuggingFace router API.
Output format follows OpenEnv spec.
"""

import os
import json
import requests
from openai import OpenAI

# Environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")
ENV_URL      = os.getenv("ENV_URL",      "http://localhost:7860")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

def env_reset(task_id: str) -> dict:
    r = requests.post(f"{ENV_URL}/reset",
                      json={"task_id": task_id},
                      headers={"Content-Type": "application/json"})
    r.raise_for_status()
    data = r.json()
    # openenv-core wraps in {"observation": {...}, "reward": ..., "done": ...}
    if "observation" in data:
        return data["observation"]
    return data

def env_step(action: dict) -> dict:
    r = requests.post(f"{ENV_URL}/step",
                      json=action,
                      headers={"Content-Type": "application/json"})
    r.raise_for_status()
    return r.json()

def build_prompt(obs: dict) -> str:
    table_str = json.dumps(obs.get("current_table", []), indent=2)
    issues_remaining = obs.get("issues_remaining", 0)
    last_result = obs.get("last_action_result", "")
    return f"""You are a data cleaning agent. Fix data quality issues in this table.

Current table:
{table_str}

Issues remaining: {issues_remaining}
Last action result: {last_result}

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
    except Exception:
        return '{"action_type": "done"}'

def parse_action(raw: str) -> dict:
    try:
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)
    except Exception:
        return {"action_type": "done"}

def run_episode(task_id: str) -> dict:
    obs = env_reset(task_id)
    rewards = []
    MAX_STEPS = 25

    print(f"[START] task={task_id} env=datacleanenv model={MODEL_NAME}")

    for step_num in range(1, MAX_STEPS + 1):
        if obs.get("done", False):
            break

        try:
            prompt = build_prompt(obs)
            raw_action = call_llm(prompt)
            action_dict = parse_action(raw_action)
            action_str = json.dumps(action_dict).replace(" ", "")

            result = env_step(action_dict)

            # Handle openenv-core response format
            if "observation" in result:
                obs = result["observation"]
                reward = result.get("reward", 0.0) or 0.0
                done = result.get("done", False)
            else:
                obs = result
                reward = obs.get("reward", 0.0) or 0.0
                done = obs.get("done", False)

            error = "null"
        except Exception as e:
            reward = 0.0
            done = False
            error = str(e)[:100]
            action_str = '{"action_type":"done"}'

        rewards.append(float(reward))
        done_str = "true" if done else "false"

        print(f"[STEP] step={step_num} action={action_str} reward={float(reward):.2f} done={done_str} error={error}")

        if done:
            break

    issues_remaining = obs.get("issues_remaining", 1)
    success = issues_remaining == 0

    # Score must be strictly in (0, 1) — clamp away from 0.0 and 1.0
    total_reward = sum(rewards)
    raw_score = total_reward / max(len(rewards), 1)
    score = max(0.001, min(0.999, raw_score))

    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(f"[END] success={success_str} steps={len(rewards)} rewards={rewards_str} score={score:.4f}")

    return {"task_id": task_id, "success": success, "steps": len(rewards), "rewards": rewards, "score": score}


if __name__ == "__main__":
    tasks = ["easy", "medium", "hard"]
    results = []

    for tid in tasks:
        try:
            res = run_episode(tid)
            results.append(res)
        except Exception as e:
            print(f"[END] success=false steps=0 rewards=0.00")
            print(f"Error in task {tid}: {e}")
        print()

    print("=" * 50)
    print("BASELINE SCORES SUMMARY")
    print("=" * 50)
    for r in results:
        total = sum(r["rewards"])
        print(f"Task: {r['task_id']:8s} | Steps: {r['steps']:3d} | Success: {str(r['success']):5s} | Total Reward: {total:.2f}")
