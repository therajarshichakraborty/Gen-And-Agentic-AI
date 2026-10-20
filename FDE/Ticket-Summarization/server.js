import dotenv from "dotenv";
import { createApp } from "./app.js";
import { createSummarizeService } from "./summarize.js";
dotenv.config();

const port = Number.parseInt(process.env.PORT || "8080");
const app = createApp(createSummarizeService());

app.listen(port, () => {
  console.log(`Server is running at the port no. : ${port}`);
});
