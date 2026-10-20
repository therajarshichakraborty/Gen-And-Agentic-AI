import OpenAI from "openai";
import { z } from "zod";
import { createInterface } from "node:readline/promises";
import { stdin, stdout, stderr } from "node:process";

const ConfigSchema = z.object({
  OPENAI_API_KEY: z.string().min(1),
  OPENAI_MODEL: z.string().min(1),
  MAX_DEPTH: z.coerce.number().int().min(1).max(5).default(3),
  BRANCHING_FACTOR: z.coerce.number().int().min(2).max(5).default(3),
  MAX_CANDIDATES: z.coerce.number().int().min(2).max(50).default(15)
});

const config = ConfigSchema.parse({
  OPENAI_API_KEY: process.env.OPENAI_API_KEY,
  OPENAI_MODEL: process.env.OPENAI_MODEL,
  MAX_DEPTH: process.env.MAX_DEPTH ?? 3,
  BRANCHING_FACTOR: process.env.BRANCHING_FACTOR ?? 3,
  MAX_CANDIDATES: process.env.MAX_CANDIDATES ?? 15
});

const client = new OpenAI({
  apiKey: config.OPENAI_API_KEY,
  timeout: 60_000,
  maxRetries: 2
});

const CandidateSchema = z.object({
  approach: z.string().min(1),
  reasoning: z.string().min(1),
  score: z.number().min(0).max(10),
  risks: z.array(z.string())
});

const ExpansionSchema = z.object({
  candidates: z.array(CandidateSchema).min(1)
});

type Candidate = z.infer<typeof CandidateSchema>;

type ThoughtNode = Readonly<{
  id: string;
  depth: number;
  parentId: string | null;
  candidate: Candidate;
  path: readonly string[];
}>;

type TreeOfThoughtsResult = Readonly<{
  best: ThoughtNode;
  explored: readonly ThoughtNode[];
  depthReached: number;
  model: string;
  usage: Readonly<{
    inputTokens: number;
    outputTokens: number;
  }>;
}>;

type TaskInput = Readonly<{
  task: string;
  context: string;
  constraints: readonly string[];
}>;

class TreeOfThoughtsError extends Error {
  constructor(
    message: string,
    public readonly cause?: unknown
  ) {
    super(message);
    this.name = "TreeOfThoughtsError";
  }
}

const parseModelJson = <T>(text: string, schema: z.ZodType<T>): T => {
  let value: unknown;

  try {
    value = JSON.parse(text);
  } catch (error) {
    throw new TreeOfThoughtsError("Model returned invalid JSON.", error);
  }

  return schema.parse(value);
};

const requestJson = async <T>(
  system: string,
  input: string,
  schema: z.ZodType<T>
): Promise<{
  data: T;
  model: string;
  inputTokens: number;
  outputTokens: number;
}> => {
  try {
    const response = await client.responses.create({
      model: config.OPENAI_MODEL,
      instructions: system,
      input,
      text: { format: { type: "json_object" } },
      max_output_tokens: 4000,
      store: false
    });

    if (response.status !== "completed") {
      throw new TreeOfThoughtsError(
        `Model response did not complete: ${response.status}`
      );
    }

    const output = response.output_text.trim();

    if (!output) {
      throw new TreeOfThoughtsError("Model returned empty output.");
    }

    return {
      data: parseModelJson(output, schema),
      model: response.model,
      inputTokens: response.usage?.input_tokens ?? 0,
      outputTokens: response.usage?.output_tokens ?? 0
    };
  } catch (error) {
    if (error instanceof TreeOfThoughtsError) {
      throw error;
    }

    if (error instanceof z.ZodError) {
      throw new TreeOfThoughtsError(
        "Model output failed schema validation.",
        error
      );
    }

    if (error instanceof OpenAI.APIError) {
      throw new TreeOfThoughtsError(
        `OpenAI API error (${error.status ?? "unknown"}): ${error.message}`,
        error
      );
    }

    throw new TreeOfThoughtsError("Model request failed.", error);
  }
};

const buildTask = (input: TaskInput): string =>
  [
    `Task:\n${input.task.trim()}`,
    input.context.trim() ? `Context:\n${input.context.trim()}` : "",
    input.constraints.length
      ? `Constraints:\n${input.constraints
          .map((item, index) => `${index + 1}. ${item.trim()}`)
          .join("\n")}`
      : ""
  ]
    .filter(Boolean)
    .join("\n\n");

const expandThoughts = async (
  task: string,
  parent: ThoughtNode | null,
  depth: number
) => {
  const result = await requestJson(
    [
      "You are the expansion stage of a Tree of Thoughts search algorithm.",
      "Generate distinct, feasible candidate approaches for the task.",
      "Evaluate each approach using correctness, feasibility, constraint satisfaction, and risk.",
      "Scores range from 0 to 10, where higher is better.",
      "Return concise candidate summaries, not private chain-of-thought.",
      "Do not claim to execute external actions.",
      "Return valid JSON matching the requested schema."
    ].join(" "),
    JSON.stringify({
      task,
      depth,
      parentApproach: parent?.candidate.approach ?? null,
      parentPath: parent?.path ?? [],
      parentRisks: parent?.candidate.risks ?? [],
      branchingFactor: config.BRANCHING_FACTOR,
      outputSchema: {
        candidates: [
          {
            approach: "string",
            reasoning: "brief rationale",
            score: "number from 0 to 10",
            risks: ["string"]
          }
        ]
      }
    }),
    ExpansionSchema
  );

  const candidates = result.data.candidates
    .slice(0, config.BRANCHING_FACTOR)
    .sort((a, b) => b.score - a.score);

  return {
    candidates,
    model: result.model,
    inputTokens: result.inputTokens,
    outputTokens: result.outputTokens
  };
};

const evaluateFinal = async (
  task: string,
  nodes: readonly ThoughtNode[]
): Promise<ThoughtNode> => {
  const result = await requestJson(
    [
      "You are a final candidate evaluator.",
      "Select the strongest candidate based on correctness, feasibility, risks, and the original constraints.",
      "Use only candidate IDs provided in the input.",
      "Return JSON with a single bestId string.",
      "Do not claim to execute external actions."
    ].join(" "),
    JSON.stringify({
      task,
      candidates: nodes.map((node) => ({
        id: node.id,
        depth: node.depth,
        approach: node.candidate.approach,
        rationale: node.candidate.reasoning,
        score: node.candidate.score,
        risks: node.candidate.risks,
        path: node.path
      })),
      outputSchema: { bestId: "string" }
    }),
    z.object({ bestId: z.string().min(1) })
  );

  const selected = nodes.find((node) => node.id === result.data.bestId);

  if (!selected) {
    throw new TreeOfThoughtsError(
      "Final evaluator selected an unknown candidate."
    );
  }

  return selected;
};

const solveWithTreeOfThoughts = async (
  input: TaskInput
): Promise<TreeOfThoughtsResult> => {
  if (!input.task.trim()) {
    throw new TreeOfThoughtsError("Task cannot be empty.");
  }

  const task = buildTask(input);
  const explored: ThoughtNode[] = [];
  let frontier: ThoughtNode[] = [];
  let sequence = 0;
  let totalInputTokens = 0;
  let totalOutputTokens = 0;
  let model = config.OPENAI_MODEL;
  let depthReached = 0;

  for (let depth = 0; depth < config.MAX_DEPTH; depth++) {
    const parents: (ThoughtNode | null)[] = depth === 0 ? [null] : frontier;

    const expanded = await Promise.all(
      parents.map(async (parent) => {
        const result = await expandThoughts(task, parent, depth + 1);

        totalInputTokens += result.inputTokens;
        totalOutputTokens += result.outputTokens;
        model = result.model;

        return { parent, candidates: result.candidates };
      })
    );

    const next: ThoughtNode[] = [];

    for (const group of expanded) {
      for (const candidate of group.candidates) {
        const parent = group.parent;
        const node: ThoughtNode = {
          id: `thought-${++sequence}`,
          depth: depth + 1,
          parentId: parent?.id ?? null,
          candidate,
          path: [...(parent?.path ?? []), candidate.approach]
        };

        next.push(node);
      }
    }

    if (next.length === 0) {
      throw new TreeOfThoughtsError("Search produced no candidates.");
    }

    const ranked = next.sort((a, b) => b.candidate.score - a.candidate.score);

    const remainingBudget = config.MAX_CANDIDATES - explored.length;

    if (remainingBudget <= 0) {
      break;
    }

    const retained = ranked.slice(0, remainingBudget);

    explored.push(...retained);
    frontier = retained;
    depthReached = depth + 1;

    if (explored.length >= config.MAX_CANDIDATES) {
      break;
    }
  }

  if (explored.length === 0) {
    throw new TreeOfThoughtsError("No candidates were explored.");
  }

  const best = await evaluateFinal(task, explored);

  return {
    best,
    explored,
    depthReached,
    model,
    usage: {
      inputTokens: totalInputTokens,
      outputTokens: totalOutputTokens
    }
  };
};

const main = async (): Promise<void> => {
  const terminal = createInterface({ input: stdin, output: stdout });

  try {
    const task = (await terminal.question("Task: ")).trim();
    const context = (await terminal.question("Context: ")).trim();
    const rawConstraints = await terminal.question(
      "Constraints separated by |: "
    );

    const result = await solveWithTreeOfThoughts({
      task,
      context,
      constraints: rawConstraints
        .split("|")
        .map((item) => item.trim())
        .filter(Boolean)
    });

    stdout.write("\nBest candidate\n\n");
    stdout.write(`${result.best.candidate.approach}\n\n`);
    stdout.write(`Rationale: ${result.best.candidate.reasoning}\n`);
    stdout.write(`Score: ${result.best.candidate.score}/10\n`);
    stdout.write(`Risks: ${JSON.stringify(result.best.candidate.risks)}\n`);

    stdout.write("\nSelected path\n");

    for (const [index, thought] of result.best.path.entries()) {
      stdout.write(`${index + 1}. ${thought}\n`);
    }

    stdout.write(`\nExplored candidates: ${result.explored.length}\n`);
    stdout.write(`Depth reached: ${result.depthReached}\n`);
    stdout.write(`Model: ${result.model}\n`);
    stdout.write(
      `Token usage: ${result.usage.inputTokens + result.usage.outputTokens}\n`
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    stderr.write(`Tree of Thoughts failed: ${message}\n`);
    process.exitCode = 1;
  } finally {
    terminal.close();
  }
};

await main();
