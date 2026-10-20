import "dotenv/config";
import OpenAI from "openai";
import { z } from "zod";
import { createInterface } from "node:readline/promises";
import { stdin, stdout, stderr } from "node:process";

const ConfigSchema = z.object({
  OPENAI_API_KEY: z.string().min(1),
  OPENAI_MODEL: z.string().min(1),
  OPENAI_TIMEOUT_MS: z.coerce.number().int().min(1000).default(60000)
});

const config = ConfigSchema.parse({
  OPENAI_API_KEY: process.env.OPENAI_API_KEY,
  OPENAI_MODEL: process.env.OPENAI_MODEL,
  OPENAI_TIMEOUT_MS: process.env.OPENAI_TIMEOUT_MS ?? 60000
});

const client = new OpenAI({
  apiKey: config.OPENAI_API_KEY,
  timeout: config.OPENAI_TIMEOUT_MS,
  maxRetries: 2
});

const StepBackSchema = z.object({
  principles: z.array(z.string()).min(1),
  solution: z.string().min(1),
  assumptions: z.array(z.string()),
  limitations: z.array(z.string())
});

type StepBackInput = Readonly<{
  task: string;
  context: string;
  constraints: readonly string[];
}>;

type StepBackResult = Readonly<{
  principles: readonly string[];
  solution: string;
  assumptions: readonly string[];
  limitations: readonly string[];
  model: string;
  responseId: string;
  usage: Readonly<{
    inputTokens: number;
    outputTokens: number;
    totalTokens: number;
  }>;
}>;

class StepBackError extends Error {
  constructor(
    message: string,
    public readonly cause?: unknown
  ) {
    super(message);
    this.name = "StepBackError";
  }
}

const buildStepBackPrompt = (input: StepBackInput): string =>
  [
    "Original task:",
    input.task.trim(),
    input.context.trim() ? `Context:\n${input.context.trim()}` : "",
    input.constraints.length
      ? `Constraints:\n${input.constraints
          .map((constraint, index) => `${index + 1}. ${constraint.trim()}`)
          .join("\n")}`
      : "",
    "First stage: Step back from the specific task. Identify the general concepts, governing principles, relevant constraints, and problem-solving approach needed to solve this class of problem. Do not solve the original task yet.",
    "Second stage: Use those principles to produce a concrete solution to the original task. Address the constraints and distinguish assumptions from known facts.",
    "Return a JSON object with principles (string array), solution (string), assumptions (string array), and limitations (string array)."
  ]
    .filter(Boolean)
    .join("\n\n");

const stepBack = async (input: StepBackInput): Promise<StepBackResult> => {
  if (!input.task.trim()) {
    throw new StepBackError("Task cannot be empty.");
  }

  if (input.constraints.some((constraint) => !constraint.trim())) {
    throw new StepBackError("Constraints cannot contain empty values.");
  }

  try {
    const response = await client.responses.create({
      model: config.OPENAI_MODEL,
      instructions: [
        "You are a careful problem-solving assistant.",
        "Apply Step-Back Prompting by identifying general principles before solving the concrete task.",
        "Keep the principles relevant and actionable.",
        "Do not fabricate facts or claim to execute external actions.",
        "Return only the requested structured output."
      ].join(" "),
      input: buildStepBackPrompt(input),
      text: {
        format: {
          type: "json_schema",
          name: "step_back_result",
          strict: true,
          schema: {
            type: "object",
            properties: {
              principles: {
                type: "array",
                items: { type: "string" }
              },
              solution: { type: "string" },
              assumptions: {
                type: "array",
                items: { type: "string" }
              },
              limitations: {
                type: "array",
                items: { type: "string" }
              }
            },
            required: ["principles", "solution", "assumptions", "limitations"],
            additionalProperties: false
          }
        }
      },
      max_output_tokens: 5000,
      store: false
    });

    if (response.status !== "completed") {
      throw new StepBackError(`Response did not complete: ${response.status}`);
    }

    const raw = response.output_text.trim();

    if (!raw) {
      throw new StepBackError("The model returned empty output.");
    }

    let parsed: unknown;

    try {
      parsed = JSON.parse(raw);
    } catch (error) {
      throw new StepBackError("Invalid JSON returned by the model.", error);
    }

    const result = StepBackSchema.parse(parsed);

    return {
      ...result,
      model: response.model,
      responseId: response.id,
      usage: {
        inputTokens: response.usage?.input_tokens ?? 0,
        outputTokens: response.usage?.output_tokens ?? 0,
        totalTokens: response.usage?.total_tokens ?? 0
      }
    };
  } catch (error) {
    if (error instanceof StepBackError) {
      throw error;
    }

    if (error instanceof z.ZodError) {
      throw new StepBackError("Model output failed validation.", error);
    }

    if (error instanceof OpenAI.APIError) {
      throw new StepBackError(
        `OpenAI API error (${error.status ?? "unknown"}): ${error.message}`,
        error
      );
    }

    throw new StepBackError("Step-Back Prompting failed.", error);
  }
};

const main = async (): Promise<void> => {
  const terminal = createInterface({ input: stdin, output: stdout });

  try {
    const task = (await terminal.question("Task: ")).trim();
    const context = (await terminal.question("Context: ")).trim();
    const rawConstraints = await terminal.question(
      "Constraints separated by |: "
    );

    const result = await stepBack({
      task,
      context,
      constraints: rawConstraints
        .split("|")
        .map((item) => item.trim())
        .filter(Boolean)
    });

    stdout.write("\nGeneral principles\n");

    for (const [index, principle] of result.principles.entries()) {
      stdout.write(`${index + 1}. ${principle}\n`);
    }

    stdout.write("\nSolution\n\n");
    stdout.write(`${result.solution}\n`);

    stdout.write("\nAssumptions\n");
    stdout.write(`${JSON.stringify(result.assumptions, null, 2)}\n`);

    stdout.write("\nLimitations\n");
    stdout.write(`${JSON.stringify(result.limitations, null, 2)}\n`);

    stdout.write(`\nModel: ${result.model}\n`);
    stdout.write(`Response ID: ${result.responseId}\n`);
    stdout.write(`Token usage: ${result.usage.totalTokens}\n`);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    stderr.write(`Error: ${message}\n`);
    process.exitCode = 1;
  } finally {
    terminal.close();
  }
};

await main();
