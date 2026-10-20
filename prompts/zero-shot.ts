import OpenAI from "openai";
import { z } from "zod";
import { createInterface } from "node:readline/promises";
import { stdin, stdout, stderr } from "node:process";

const ConfigSchema = z.object({
  OPENAI_API_KEY: z.string().min(1),
  OPENAI_MODEL: z.string().min(1)
});

const config = ConfigSchema.parse({
  OPENAI_API_KEY: process.env.OPENAI_API_KEY,
  OPENAI_MODEL: process.env.OPENAI_MODEL
});

const client = new OpenAI({
  apiKey: config.OPENAI_API_KEY,
  timeout: 60_000,
  maxRetries: 2
});

const OutputSchema = z.object({
  answer: z.string().min(1),
  confidence: z.enum(["low", "medium", "high"]),
  assumptions: z.array(z.string()),
  limitations: z.array(z.string())
});

type ZeroShotInput = Readonly<{
  task: string;
  context: string;
  constraints: readonly string[];
  systemInstructions: string;
}>;

type ZeroShotResult = Readonly<{
  output: z.infer<typeof OutputSchema>;
  model: string;
  responseId: string;
  usage: Readonly<{
    inputTokens: number;
    outputTokens: number;
    totalTokens: number;
  }>;
}>;

class ZeroShotError extends Error {
  constructor(
    message: string,
    public readonly cause?: unknown
  ) {
    super(message);
    this.name = "ZeroShotError";
  }
}

const buildPrompt = (input: ZeroShotInput): string => {
  const sections = [
    `Task:\n${input.task.trim()}`,
    input.context.trim() ? `Context:\n${input.context.trim()}` : "",
    input.constraints.length
      ? `Constraints:\n${input.constraints
          .map((constraint, index) => `${index + 1}. ${constraint.trim()}`)
          .join("\n")}`
      : "",
    [
      "Requirements:",
      "Solve the task directly without relying on supplied examples.",
      "Do not invent facts or conceal uncertainty.",
      "State material assumptions and limitations.",
      "Return a complete, self-contained answer."
    ].join("\n")
  ];

  return sections.filter(Boolean).join("\n\n");
};

const zeroShot = async (input: ZeroShotInput): Promise<ZeroShotResult> => {
  if (!input.task.trim()) {
    throw new ZeroShotError("Task cannot be empty.");
  }

  if (input.constraints.some((constraint) => !constraint.trim())) {
    throw new ZeroShotError("Constraints cannot contain empty values.");
  }

  const prompt = buildPrompt(input);

  try {
    const response = await client.responses.create({
      model: config.OPENAI_MODEL,
      instructions:
        input.systemInstructions.trim() ||
        "You are a precise assistant. Follow the task requirements.",
      input: prompt,
      text: {
        format: {
          type: "json_schema",
          name: "zero_shot_result",
          strict: true,
          schema: {
            type: "object",
            properties: {
              answer: { type: "string" },
              confidence: {
                type: "string",
                enum: ["low", "medium", "high"]
              },
              assumptions: {
                type: "array",
                items: { type: "string" }
              },
              limitations: {
                type: "array",
                items: { type: "string" }
              }
            },
            required: ["answer", "confidence", "assumptions", "limitations"],
            additionalProperties: false
          }
        }
      },
      max_output_tokens: 4000,
      store: false
    });

    if (response.status !== "completed") {
      throw new ZeroShotError(`Response did not complete: ${response.status}`);
    }

    const rawOutput = response.output_text.trim();

    if (!rawOutput) {
      throw new ZeroShotError("The model returned empty output.");
    }

    let parsed: unknown;

    try {
      parsed = JSON.parse(rawOutput);
    } catch (error) {
      throw new ZeroShotError("The model returned invalid JSON.", error);
    }

    const output = OutputSchema.parse(parsed);

    return {
      output,
      model: response.model,
      responseId: response.id,
      usage: {
        inputTokens: response.usage?.input_tokens ?? 0,
        outputTokens: response.usage?.output_tokens ?? 0,
        totalTokens: response.usage?.total_tokens ?? 0
      }
    };
  } catch (error) {
    if (error instanceof ZeroShotError) {
      throw error;
    }

    if (error instanceof z.ZodError) {
      throw new ZeroShotError("Output failed runtime validation.", error);
    }

    if (error instanceof OpenAI.APIError) {
      throw new ZeroShotError(
        `OpenAI API error (${error.status ?? "unknown"}): ${error.message}`,
        error
      );
    }

    throw new ZeroShotError("Zero-shot generation failed.", error);
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
    const systemInstructions = (
      await terminal.question("System instructions (optional): ")
    ).trim();

    const result = await zeroShot({
      task,
      context,
      constraints: rawConstraints
        .split("|")
        .map((item) => item.trim())
        .filter(Boolean),
      systemInstructions
    });

    stdout.write("\nAnswer\n\n");
    stdout.write(`${result.output.answer}\n\n`);
    stdout.write(`Confidence: ${result.output.confidence}\n`);
    stdout.write(`Assumptions: ${JSON.stringify(result.output.assumptions)}\n`);
    stdout.write(`Limitations: ${JSON.stringify(result.output.limitations)}\n`);
    stdout.write(`Model: ${result.model}\n`);
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
