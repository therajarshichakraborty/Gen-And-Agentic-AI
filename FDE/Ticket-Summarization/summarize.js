import OpenAI from "openai";

const default_model = "gpt-5.6-luna";

export function createSummarizeService({
  client = new OpenAI(),
  model = process.env.OPENAI_MODEL || default_model
} = {}) {
  return {
    async summarize(ticket) {
      const response = await client.responses.create({
        model,
        input: `Summarize this support ticket in 2 lines : \n\n${ticket}`,
        store: false
      });

      if (!response.output_text) {
        throw new Error("OpenAI returned an empty summary");
      }

      return response.output_text;
    }
  };
}
