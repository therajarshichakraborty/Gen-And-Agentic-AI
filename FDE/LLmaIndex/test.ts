import LlamaCloud from "@llamaindex/llama-cloud";
import fs from "node:fs";

const client = new LlamaCloud();

const file = await client.files.create({
  file: fs.createReadStream("document.pdf"),
  purpose: "parse"
});
const result = await client.parsing.parse({
  file_id: file.id,
  tier: "agentic",
  version: "latest",
  expand: ["markdown"]
});

const response = (result as any).markdown.pages[0].markdown;
console.log(response);
