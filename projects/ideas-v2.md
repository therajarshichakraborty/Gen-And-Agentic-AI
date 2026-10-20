# Top 3 Full-Stack Applied AI Projects

These three projects are designed to cover the most important applied AI engineering topics through practical, production-oriented applications. Together, they cover LLM APIs, prompt engineering, structured outputs, RAG, vector search, agents, tool calling, human-in-the-loop workflows, evaluation, guardrails, observability, and deployment.

## 1. KnowledgeOS: Enterprise Knowledge and Research Platform

### Overview

Build a workspace where users upload documents, connect knowledge sources, ask questions, and receive grounded answers with citations. The system should handle complex research questions by planning searches, retrieving evidence, and synthesizing a final report.

### Core features

- Workspace and organization management
- PDF, DOCX, Markdown, TXT, and URL ingestion
- Document parsing, cleaning, chunking, and metadata extraction
- Background ingestion jobs with progress tracking
- Hybrid search using dense embeddings and BM25
- Metadata filters and reranking
- RAG answers with citations linked to source passages
- Query rewriting, multi-query retrieval, and context compression
- Research mode that breaks a question into subquestions
- Report generation with source references
- Conversation history and persistent user preferences
- Document permissions and tenant-level data isolation
- Feedback controls for answers and citations
- Admin dashboard for ingestion health, quality, usage, and costs

### Applied AI topics covered

- OpenAI, Anthropic, or Gemini APIs
- Prompt templates and prompt versioning
- Zero-shot and few-shot prompting
- Structured outputs with Pydantic or Zod
- Embeddings and semantic search
- Chunking strategies and metadata extraction
- Vector databases and PostgreSQL with pgvector
- BM25 and hybrid retrieval
- Query rewriting and multi-query retrieval
- Reranking and context compression
- RAG and advanced RAG patterns
- Citation generation and grounded answers
- Conversation memory and context engineering
- Human feedback collection
- RAG evaluation using golden datasets
- Faithfulness, answer relevance, context precision, and context recall
- Prompt injection defense for uploaded documents
- PII handling and access control
- Tracing, token usage, latency, and cost monitoring

### Suggested stack

- Frontend: Next.js, TypeScript, Tailwind CSS, shadcn/ui
- Backend: FastAPI or Next.js route handlers
- Database: PostgreSQL
- Vector search: pgvector or Qdrant
- Cache and queues: Redis and BullMQ, or Celery with Redis
- AI: OpenAI SDK with a provider abstraction
- Evaluation and tracing: Langfuse or LangSmith
- Storage: S3-compatible object storage
- Deployment: Docker and a cloud platform

### Production-grade requirements

- Run ingestion asynchronously instead of blocking requests.
- Make ingestion jobs idempotent and retryable.
- Enforce document permissions during retrieval, not only in the UI.
- Store source metadata and stable passage identifiers for citations.
- Version prompts, chunking configurations, and embedding models.
- Build a repeatable evaluation dataset before changing retrieval settings.
- Track latency and cost per successful answer.
- Add deletion workflows that remove both stored files and indexed chunks.

### Definition of done

A user can upload a collection of documents, ask a difficult question, receive a cited answer, inspect the exact supporting passages, provide feedback, and view usage and quality metrics.

---

## 2. AgentDesk: AI Operations and Workflow Automation Platform

### Overview

Build a platform where users describe a task in natural language and an AI agent plans and executes it using approved tools. Examples include preparing a CRM report, triaging support tickets, drafting follow-up emails, researching a vendor, or generating a weekly project summary.

The platform must distinguish between safe read operations and actions that change external systems. Sensitive actions require human approval.

### Core features

- Workflow builder with reusable steps
- Natural-language task input
- Planner and executor workflow
- Tool registry with typed input and output schemas
- Integrations for email, calendar, issue tracking, CRM, and web search
- Scheduled and event-triggered workflows
- Human approval inbox for sensitive actions
- Pause, resume, cancel, retry, and replay controls
- Execution timeline showing model decisions and tool results
- Workflow templates and reusable agent skills
- Persistent state for long-running workflows
- Notifications when approval or user input is required
- Role-based access control and audit history
- Usage quotas and per-workspace cost tracking

### Example workflow

1. A user asks the system to research a vendor and prepare a recommendation.
2. The planner creates a sequence of research tasks.
3. Search tools collect information from permitted sources.
4. The agent summarizes evidence and creates a structured recommendation.
5. The system drafts an email but does not send it automatically.
6. A human reviews the draft and approves or revises it.
7. The workflow sends the approved email and records the result.

### Applied AI topics covered

- Tool calling and function calling
- Structured outputs and schema validation
- ReAct-style agents
- Planner-executor and orchestrator-worker patterns
- LangGraph state graphs
- Conditional routing and parallel execution
- Agent memory and persistent state
- MCP clients and servers
- Tool selection and permission boundaries
- Human-in-the-loop approval and revision
- Confidence-based escalation
- Retries, fallbacks, and error recovery
- Checkpointing and durable execution
- Idempotency for external actions
- Prompt injection defense for web pages and tool outputs
- Sandboxed code execution where needed
- Agent trajectory evaluation
- End-to-end task success metrics
- Trace inspection and audit logs
- Cost limits and model routing

### Suggested stack

- Frontend: Next.js, TypeScript, Tailwind CSS, shadcn/ui
- Backend: FastAPI or Node.js with TypeScript
- Agent orchestration: LangGraph
- Database: PostgreSQL
- Queue and cache: Redis with BullMQ, or Celery
- Integrations: OAuth-based provider APIs and MCP
- AI: OpenAI SDK, with optional alternate providers
- Observability: Langfuse or LangSmith plus OpenTelemetry
- Deployment: Docker, CI/CD, and a cloud platform

### Production-grade requirements

- Require explicit approval for destructive, external, or irreversible actions.
- Validate every tool argument against a strict schema.
- Keep authorization checks in the backend tool layer.
- Use idempotency keys for actions such as sending messages or creating tickets.
- Persist workflow state so executions can resume after a restart.
- Apply per-step timeouts, retry limits, and overall task budgets.
- Treat retrieved web content and tool outputs as untrusted input.
- Record who approved each action and what data was sent.
- Provide a dry-run mode for testing workflows safely.

### Definition of done

A user can create a multi-step workflow, watch its execution, inspect each tool call, approve a sensitive action, resume the workflow, and review a complete audit trail.

---

## 3. EvalForge: AI Evaluation, Testing, and Observability Platform

### Overview

Build a platform for developers to test and improve LLM applications before and after deployment. Users can create datasets, compare prompts and models, evaluate RAG pipelines, test agent trajectories, run adversarial cases, and monitor production quality.

This project is especially valuable because reliable evaluation is essential for shipping every other AI application.

### Core features

- Projects for different AI applications
- Dataset creation, import, versioning, and labeling
- Test cases with expected answers, sources, or structured outputs
- Side-by-side prompt and model comparisons
- Batch evaluation runs
- LLM-as-a-judge and deterministic evaluators
- RAG retrieval and answer quality metrics
- Agent tool-call and trajectory evaluation
- Regression tests for prompts and model changes
- Prompt and model version history
- Adversarial test suites for prompt injection and data leakage
- Traces for requests, model calls, retrieval, and tool execution
- Token, latency, error, and cost analytics
- Threshold-based alerts for quality regressions
- Feedback ingestion from production applications
- Reports that identify failing examples and likely causes

### Applied AI topics covered

- Prompt engineering and prompt versioning
- Structured outputs and validation
- LLM-as-a-judge
- Golden datasets and human-labeled datasets
- Offline and online evaluation
- Regression testing and A/B testing
- RAGAS or equivalent RAG evaluation
- Faithfulness and answer relevance
- Context precision and context recall
- Retrieval precision, recall, and ranking metrics
- Tool-call accuracy and agent trajectory evaluation
- Hallucination and groundedness checks
- Adversarial testing and prompt injection evaluation
- Model comparison and model routing
- Token accounting and cost estimation
- Latency and reliability monitoring
- Tracing and distributed observability
- Dataset and experiment versioning
- Feedback loops and continuous evaluation

### Suggested stack

- Frontend: Next.js, TypeScript, Tailwind CSS, shadcn/ui
- Backend: FastAPI or Node.js with TypeScript
- Database: PostgreSQL
- Background jobs: Redis with BullMQ, or Celery
- Storage: S3-compatible object storage for larger datasets and artifacts
- Evaluation: Python, Pydantic, and custom evaluators
- AI providers: OpenAI SDK plus optional alternate providers
- Observability: OpenTelemetry and Langfuse
- Deployment: Docker and CI/CD

### Production-grade requirements

- Version datasets, prompts, evaluator definitions, and model settings.
- Make evaluation runs reproducible by storing configuration and provider metadata.
- Separate deterministic metrics from subjective model-judge scores.
- Calibrate model judges against human-labeled examples.
- Track evaluator cost and avoid unnecessary judge calls.
- Support retries without duplicating results.
- Protect datasets that contain private or sensitive information.
- Set thresholds that block releases when critical regression tests fail.
- Keep raw traces access-controlled and apply retention policies.

### Definition of done

A developer can upload a dataset, run the same tests against two prompt or model versions, compare quality, latency, and cost, inspect failed examples, and prevent a release from passing when required thresholds are not met.

---

## Shared engineering foundation

Use the following foundation across all three projects:

- Authentication and organization-level authorization
- PostgreSQL migrations and clear data ownership
- Environment validation and secret management
- Typed request and response schemas
- Structured logging and request IDs
- Rate limiting and usage quotas
- Retries with exponential backoff
- Timeouts and graceful failure handling
- Background jobs for long-running work
- Docker-based local development
- Automated tests and CI/CD
- Health checks and production monitoring
- Database indexes and query optimization
- Data retention and deletion workflows
- Secure handling of API keys and user data
- Cost and latency budgets
- A README with architecture, setup, tradeoffs, and demo instructions

## Recommended build order

1. **KnowledgeOS first.** It teaches ingestion, embeddings, vector search, retrieval, RAG, citations, and evaluation.
2. **AgentDesk second.** Reuse your AI and backend foundations to learn stateful agents, tool execution, workflow orchestration, and human approval.
3. **EvalForge third.** Build evaluation and observability capabilities that can test KnowledgeOS and AgentDesk.

## Portfolio strategy

Do not build all three as shallow demos. Build one complete project at a time. Each should include an architecture diagram, a deployed demo, seed data, automated tests, evaluation results, a cost and latency report, and clear documentation of failure cases and tradeoffs.

Together, these projects demonstrate applied AI engineering beyond basic chat interfaces: retrieval systems, reliable agents, human oversight, measurable quality, and production operations.
