import { env } from "../env.js";

interface ThoughtStep {
  step: number;
  thought: string;
  action: string;
}

interface StepResult {
  thought: string;
  action: string;
  nextContext: string;
  isDone: boolean;
}

async function chainOfThought(initialPrompt: string) {
  let stepCount = 0;
  const maxSteps = 100;
  let isComplete = false;
  let currentContext = initialPrompt;
  const thoughtChain: ThoughtStep[] = [];

  while (!isComplete && stepCount < maxSteps) {
    stepCount++;

    const thoughtResult = await stepReasoning(currentContext, stepCount);

    thoughtChain.push({
      step: stepCount,
      thought: thoughtResult.thought,
      action: thoughtResult.action
    });

    currentContext = thoughtResult.nextContext;

    if (thoughtResult.isDone || stepCount >= maxSteps) {
      isComplete = true;
    }
  }

  return {
    finalOutput: currentContext,
    steps: thoughtChain,
    stepCounted: stepCount
  };
}

async function stepReasoning(ctx: string, step: number): Promise<StepResult> {
  console.log(`[Step ${step}] Analyzing: ${ctx}`);

  const schema = {
    name: "cot_step",
    strict: true,
    schema: {
      type: "object",
      properties: {
        thought: {
          type: "string",
          description:
            "Internal thoughts breaking down the current state of the problem."
        },
        action: {
          type: "string",
          description:
            "Action identifier, e.g., 'CALCULATE', 'FORMAT_ANSWER', or 'NONE'."
        },
        nextContext: {
          type: "string",
          description:
            "The concrete progression or updated state summarizing everything solved up to this point."
        },
        isDone: {
          type: "boolean",
          description:
            "Set to true ONLY if you have reached the definitive ultimate answer."
        }
      },
      required: ["thought", "action", "nextContext", "isDone"],
      additionalProperties: false
    }
  };

  try {
    const response = await fetch("https://openai.com", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${env.OPENAI_API_KEY}`
      },
      body: JSON.stringify({
        model: "gpt-4o-mini",
        messages: [
          {
            role: "system",
            content:
              "You are a chain-of-thought agent. Review the current context and advance the solution by exactly one incremental step. Update the context to reflect your new progress. Set isDone to true only when the problem is fully resolved."
          },
          { role: "user", content: `Current Context: ${ctx}` }
        ],
        response_format: {
          type: "json_schema",
          json_schema: schema
        },
        temperature: 0.2
      })
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`OpenAI API Error: ${response.status} - ${errorText}`);
    }

    const data = await response.json();
    const rawChoice = (data as any).choices[0].message.content;

    return JSON.parse(rawChoice) as StepResult;
  } catch (error) {
    console.error(`Error at step ${step}:`, error);
    return {
      thought: "An error occurred during API execution.",
      action: "ERROR",
      nextContext: ctx,
      isDone: true
    };
  }
}

const prompt =
  "A box contains 3 red balls and 7 blue balls. If you draw 2 balls without replacement, what is the probability that both are red? Break it down step by step.";

(async () => {
  console.log("Starting Chain of Thought process...");

  const result = await chainOfThought(prompt);

  console.log(`\n\nTotal Steps Taken: ${result.stepCounted}`);
  console.log(`\n\nFinal Output:\n${result.finalOutput}`);
})();
