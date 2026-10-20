import { query } from "@anthropic-ai/claude-agent-sdk";

const provider = (process.env.LLM_PROVIDER ||
  (process.env.OPENROUTER_API_KEY ? "openrouter" : "anthropic")) as
  "openrouter" | "anthropic";

function getProviderConfig(provider: "openrouter" | "anthropic") {
  if (provider === "openrouter") {
    const apiKey = process.env.OPENROUTER_API_KEY;
    if (!apiKey) {
      throw new Error(
        "Missing OPENROUTER_API_KEY in environment variables or .env file."
      );
    }

    return {
      provider: "openrouter" as const,
      model: process.env.MODEL || "moonshotai/kimi-k2",
      env: {
        ...process.env,
        ANTHROPIC_BASE_URL: "https://openrouter.ai/api",
        ANTHROPIC_AUTH_TOKEN: apiKey,
        OPENROUTER_API_KEY: apiKey,
        ANTHROPIC_API_KEY: ""
      }
    };
  } else {
    const apiKey = process.env.ANTHROPIC_API_KEY;
    if (!apiKey) {
      throw new Error(
        "Missing ANTHROPIC_API_KEY in environment variables or .env file."
      );
    }

    return {
      provider: "anthropic" as const,
      model: process.env.MODEL || "claude-3-7-sonnet-20250219",
      env: {
        ...process.env,
        ANTHROPIC_API_KEY: apiKey
      }
    };
  }
}

async function runAgent() {
  const config = getProviderConfig(provider);
  console.log(
    `Starting Claude Agent using [${config.provider.toUpperCase()}] provider with model: ${config.model}\n`
  );

  try {
    const agentQuery = query({
      prompt:
        "please create a lib/myTokenizer.ts file in my root which should contain a basic typescript llm tokenizer takes a string input using readline and tokenizes it.",
      options: {
        model: config.model,
        allowedTools: ["Read", "Edit", "Glob"],
        permissionMode: "acceptEdits",
        thinking: { type: "disabled" },
        env: config.env
      }
    });

    for await (const message of agentQuery) {
      if (message.type === "assistant" && message.message?.content) {
        for (const block of message.message.content) {
          if ("text" in block) {
            console.log(block.text);
          } else if ("name" in block) {
            console.log(`🔧 Tool: ${block.name}`);
          }
        }
      } else if (message.type === "result") {
        console.log(`\nDone: ${message.subtype}`);
      }
    }
  } catch (error) {
    console.error("❌ Agent error:", error);
  }
}

runAgent();
