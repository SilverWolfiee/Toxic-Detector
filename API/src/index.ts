import { Elysia, t } from "elysia";
import { cors } from "@elysiajs/cors";

const API_URL = "http://localhost:8000"; // Python inference server

const app = new Elysia()
  .use(cors())
  .get("/", () => ({ status: "ok", service: "ChatGuard Toxicity API" }))
  .post(
    "/v1/predict",
    async ({ body }) => {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: body.text }),
      });

      if (!res.ok) throw new Error("Inference server error");

      const data = await res.json();
      const score: number = data.score;

     
      let category: string;
      let action: string;

      if (score <= 0.20) {
        category = "Safe";
        action   = "allow";
      } else if (score <= 0.45) {
        category = "Ambiguous";
        action   = "warn";
      } else if (score <= 0.75) {
        category = "Implicit/Covert Hate";
        action   = "delete";
      } else {
        category = "Explicit Toxicity";
        action   = "delete";
      }

      return {
        text:     body.text,
        score,
        category,
        action,
        flagged:  score > 0.45,
      };
    },
    {
      body: t.Object({
        text:      t.String(),
        threshold: t.Optional(t.Number()),
      }),
    }
  )
  .listen(3000);

console.log(`ChatGuard API running on http://localhost:${app.server?.port}`);
