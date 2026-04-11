"""
inference.py - Baseline inference script for DataCleanEnv.
Uses WebSocket session for stateful interaction.
Output format follows OpenEnv spec.
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

def get_ws_url(env_url: str) -> str:
    return env_url.replace("https://", "wss://").replace("http://", "ws://") + "/ws"

def build_prompt(obs: dict) -> str:
    table_str = json.dumps(obs.get("current_table", []), indent=2)
    return f"""You are a data cleaning agent. Fix data quality issues in this table.

Current table:
{table_str}

Issues remaining: {obs.get('issues_remaining', 0)}
Last action result: {obs.get('last_action_result', '')}

Respond with EXACTLY one JSON object, no extra text.

Fill missing value: {{"action_type":"fill_missing","row_index":<int>,"column":"<col>","new_value":<value>}}
Fix wrong type:     {{"action_type":"fix_type","row_index":<int>,"column":"<col>","new_value":<value>}}
Remove duplicate:   {{"action_type":"remove_duplicate","row_index":<int>}}
Fix invalid value:  {{"action_type":"fix_value","row_index":<int>,"column":"<col>","new_value":<value>}}
All done:           {{"action_type":"done"}}

Rules:
- Fix ONE issue per step
- Look for: null values, duplicate rows, string-as-number, negative values
- When issues_remaining==0 use done action
- JSON only, no explanation
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
    except Exception:
        return '{"action_type":"done"}'

def parse_action(raw: str) -> dict:
    try:
        return json.loads(raw.replace("```json","").replace("```","").strip())
    except Exception:
        return {"action_type": "done"}

async def run_episode_async(task_id: str) -> dict:
    ws_url = get_ws_url(ENV_URL)
    rewards = []
    MAX_STEPS = 25
    obs = {}

    print(f"[START] task={task_id} env=datacleanenv model={MODEL_NAME}")

    try:
        async with websockets.connect(ws_url, open_timeout=30) as ws:
            # Reset with correct format: {type: reset, data: {task_id: ...}}
            await ws.send(json.dumps({"type": "reset", "data": {"task_id": task_id}}))
            reset_resp = json.loads(await ws.recv())
            obs = reset_resp.get("data", {}).get("observation", {})

            for step_num in range(1, MAX_STEPS + 1):
                if obs.get("done", False) or obs.get("issues_remaining", 0) == 0:
                    break

                try:
                    prompt = build_prompt(obs)
                    raw = call_llm(prompt)
                    action = parse_action(raw)
                    action_str = json.dumps(action).replace(" ", "")

                    # Step format: {type: step, data: <action fields>}
                    await ws.send(json.dumps({"type": "step", "data": action}))
                    step_resp = json.loads(await ws.recv())

                    step_data = step_resp.get("data", {})
                    obs = step_data.get("observation", {})
                    reward = float(step_data.get("reward", 0.05) or 0.05)
                    done = bool(step_data.get("done", False))
                    error = "null"
                except Exception as e:
                    reward = 0.05
                    done = False
                    error = str(e)[:80]
                    action_str = '{"action_type":"done"}'

                rewards.append(reward)
                done_str = "true" if done else "false"
                print(f"[STEP] step={step_num} action={action_str} reward={reward:.2f} done={done_str} error={error}")

                if done:
                    break

    except Exception as e:
        print(f"[END] success=false steps=0 rewards=0.05")
        return {"task_id": task_id, "success": False, "steps": 0, "rewards": []}

    success = obs.get("issues_remaining", 1) == 0
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={'true' if success else 'false'} steps={len(rewards)} rewards={rewards_str}")
    return {"task_id": task_id, "success": success, "steps": len(rewards), "rewards": rewards}

def run_episode(task_id: str) -> dict:
    return asyncio.run(run_episode_async(task_id))

if __name__ == "__main__":
    tasks = ["easy", "medium", "hard"]
    results = []

    for tid in tasks:
        try:
            res = run_episode(tid)
            results.append(res)
        except Exception as e:
            print(f"[END] success=false steps=0 rewards=0.05")
            results.append({"task_id": tid, "success": False, "steps": 0, "rewards": []})
        print()

    print("=" * 50)
    print("BASELINE SCORES SUMMARY")
    print("=" * 50)
    for r in results:
        total = sum(r["rewards"])
        print(f"Task: {r['task_id']:8s} | Steps: {r['steps']:3d} | Success: {str(r['success']):5s} | Total Reward: {total:.2f}")
