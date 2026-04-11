"""
inference.py - FINAL FIX (Phase 2 compliant)
"""

import os
import json
import asyncio
import websockets
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ✅ CLAMP
def clamp_reward(r):
    try:
        r = float(r)
    except:
        return 0.02
    return max(0.01, min(0.99, r))


def get_ws_url(env_url):
    return env_url.replace("https://", "wss://").replace("http://", "ws://") + "/ws"


def build_prompt(obs):
    return json.dumps(obs)


def call_llm(prompt):
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


def parse_action(raw):
    try:
        return json.loads(raw.replace("```json","").replace("```","").strip())
    except:
        return {"action_type": "done"}


async def run_episode_async(task_id):

    ws_url = get_ws_url(ENV_URL)
    rewards = []
    obs = {}
    MAX_STEPS = 25

    # ✅ REQUIRED START PRINT
    print(f"[START] task={task_id}", flush=True)

    try:
        async with websockets.connect(ws_url, open_timeout=30) as ws:

            await ws.send(json.dumps({"type": "reset", "data": {"task_id": task_id}}))
            reset_resp = json.loads(await ws.recv())
            obs = reset_resp.get("data", {}).get("observation", {})

            for step_num in range(1, MAX_STEPS + 1):

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

                # ✅ REQUIRED STEP PRINT
                print(f"[STEP] step={step_num} reward={reward:.2f} done={'true' if done else 'false'}", flush=True)

                if done:
                    break

    except:
        print(f"[END] task={task_id} score=0.02 steps=0", flush=True)
        return {
            "task_id": task_id,
            "success": False,
            "steps": 0,
            "rewards": [0.02],
            "score": 0.02
        }

    success = obs.get("issues_remaining", 1) == 0

    score = sum(rewards) / max(len(rewards), 1)
    score = clamp_reward(score)

    # ✅ REQUIRED END PRINT
    print(f"[END] task={task_id} score={score:.2f} steps={len(rewards)}", flush=True)

    return {
        "task_id": task_id,
        "success": success,
        "steps": len(rewards),
        "rewards": rewards,
        "score": score
    }


def run_episode(task_id):
    return asyncio.run(run_episode_async(task_id))


if __name__ == "__main__":
    tasks = ["easy", "medium", "hard"]

    for t in tasks:
        try:
            run_episode(t)
        except:
            print(f"[END] task={t} score=0.02 steps=0", flush=True)
