"""
inference.py - Fixed for Phase 2 validation
"""

import os
import json
import asyncio
import websockets
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")
ENV_URL      = os.getenv("ENV_URL",      "http://localhost:7860")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ✅ CLAMP FUNCTION (CRITICAL)
def clamp_reward(r):
    try:
        r = float(r)
    except:
        return 0.02

    if r <= 0.0:
        return 0.01
    elif r >= 1.0:
        return 0.99
    return r


def get_ws_url(env_url: str) -> str:
    return env_url.replace("https://", "wss://").replace("http://", "ws://") + "/ws"


def build_prompt(obs: dict) -> str:
    table_str = json.dumps(obs.get("current_table", []), indent=2)
    return f"""You are a data cleaning agent.

Fix ONE issue per step.

Table:
{table_str}

Issues remaining: {obs.get('issues_remaining', 0)}

Return ONLY JSON.

Examples:
{{"action_type":"fill_missing","row_index":1,"column":"age","new_value":32}}
{{"action_type":"remove_duplicate","row_index":3}}
{{"action_type":"done"}}
"""


def call_llm(prompt: str) -> str:
    try:
        r = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
        )
        return r.choices[0].message.content.strip()
    except:
        return '{"action_type":"done"}'


def parse_action(raw: str) -> dict:
    try:
        return json.loads(raw.replace("```json","").replace("```","").strip())
    except:
        return {"action_type": "done"}


async def run_episode_async(task_id: str) -> dict:
    ws_url = get_ws_url(ENV_URL)
    rewards = []
    MAX_STEPS = 25
    obs = {}

    try:
        async with websockets.connect(ws_url, open_timeout=30) as ws:

            await ws.send(json.dumps({"type": "reset", "data": {"task_id": task_id}}))
            reset_resp = json.loads(await ws.recv())
            obs = reset_resp.get("data", {}).get("observation", {})

            for _ in range(MAX_STEPS):

                if obs.get("done") or obs.get("issues_remaining", 0) == 0:
                    break

                try:
                    prompt = build_prompt(obs)
                    raw = call_llm(prompt)
                    action = parse_action(raw)

                    await ws.send(json.dumps({"type": "step", "data": action}))
                    step_resp = json.loads(await ws.recv())

                    step_data = step_resp.get("data", {})
                    obs = step_data.get("observation", {})

                    reward = clamp_reward(step_data.get("reward", 0.02))
                    done = bool(step_data.get("done", False))

                except:
                    reward = 0.02
                    done = False

                rewards.append(reward)

                if done:
                    break

    except:
        return {
            "task_id": task_id,
            "success": False,
            "steps": 0,
            "rewards": [0.02],
            "score": 0.02
        }

    success = obs.get("issues_remaining", 1) == 0

    # ✅ FINAL SCORE (MANDATORY)
    score = sum(rewards) / max(len(rewards), 1)
    score = clamp_reward(score)

    return {
        "task_id": task_id,
        "success": success,
        "steps": len(rewards),
        "rewards": rewards,
        "score": score
    }


def run_episode(task_id: str) -> dict:
    return asyncio.run(run_episode_async(task_id))


if __name__ == "__main__":
    tasks = ["easy", "medium", "hard"]
    results = [run_episode(t) for t in tasks]

    print("\nSUMMARY:")
    for r in results:
        print(r)
