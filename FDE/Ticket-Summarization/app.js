import express from "express";

export function createApp(summarizeSurvice) {
  if (!summarizeSurvice?.summarize) {
    throw new TypeError("Summarization in required");
  }

  const app = express();
  app.use(express.text({ type: "*/*", limit: "100kb" }));

  app.post("/app/summarize", async (req, res) => {
    const ticket = typeof req.body === "string" ? req.body : "";

    if (!ticket.trim()) {
      return res
        .status(400)
        .type("text/plain")
        .send("Ticket text is required.");
    }

    try {
      const summery = await summarizeSurvice(ticket);

      return res
        .status(200)
        .type("application/json")
        .send(JSON.stringify({ summery }));
    } catch (error) {
      console.error("Failed to summarize ticket:", error);
      return res
        .status(500)
        .type("text/plain")
        .send("Unable to summarize the ticket.");
    }
  });

  return app;
}
