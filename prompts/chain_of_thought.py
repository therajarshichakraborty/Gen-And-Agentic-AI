import json
import urllib.request
import urllib.error
from typing import List, Dict, Any
from .. import env

OPENAI_API_KEY =  env.OPENAI_API_KEY

def chain_of_thought(initial_prompt: str) -> Dict[str, Any]:
    step_count = 0
    max_steps = 100
    is_complete = False
    current_context = initial_prompt
    thought_chain: List[Dict[str, Any]] = []

    while not is_complete and step_count < max_steps:
        step_count += 1

        thought_result = step_reasoning(current_context, step_count)

        thought_chain.append({
            "step": step_count,
            "thought": thought_result["thought"],
            "action": thought_result["action"]
        })

        current_context = thought_result["nextContext"]

        if thought_result["isDone"] or step_count >= max_steps:
            is_complete = True

    return {
        "finalOutput": current_context,
        "steps": thought_chain,
        "stepCounted": step_count
    }

def step_reasoning(ctx: str, step: int) -> Dict[str, Any]:
    print(f"[Step {step}] Analyzing: {ctx[:60]}...")

    schema = {
        "name": "cot_step",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "thought": { 
                    "type": "string", 
                    "description": "Internal thoughts breaking down the current state of the problem." 
                },
                "action": { 
                    "type": "string", 
                    "description": "Action identifier, e.g., 'CALCULATE', 'FORMAT_ANSWER', or 'NONE'." 
                },
                "nextContext": { 
                    "type": "string", 
                    "description": "The concrete progression or updated state summarizing everything solved up to this point." 
                },
                "isDone": { 
                    "type": "boolean", 
                    "description": "Set to true ONLY if you have reached the definitive ultimate answer." 
                }
            },
            "required": ["thought", "action", "nextContext", "isDone"],
            "additionalProperties": False
        }
    }

    url = "https://openai.com"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            { 
                "role": "system", 
                "content": "You are a chain-of-thought agent. Review the current context and advance the solution by exactly one incremental step. Update the context to reflect your new progress. Set isDone to true only when the problem is fully resolved." 
            },
            {"role": "user", "content": f"Current Context: {ctx}"}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": schema
        },
        "temperature": 0.2
    }

    try:
        req = urllib.request.Request(
            url, 
            data=json.dumps(payload).encode("utf-8"), 
            headers=headers, 
            method="POST"
        )
        
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            raw_choice = res_data["choices"][0]["message"]["content"]
            return json.loads(raw_choice)

    except Exception as e:
        print(f"Error at step {step}: {e}")
        return {
            "thought": "An error occurred during API execution.",
            "action": "ERROR",
            "nextContext": ctx,
            "isDone": True
        }

if __name__ == "__main__":
    prompt = "A box contains 3 red balls and 7 blue balls. If you draw 2 balls without replacement, what is the probability that both are red? Break it down step by step."
    
    print("Starting Chain of Thought process...")
    result = chain_of_thought(prompt)
    
    print(f"\n\nTotal Steps Taken: {result['stepCounted']}")
    print(f"\n\nFinal Output:\n{result['finalOutput']}")
