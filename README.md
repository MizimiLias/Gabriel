# Gabriel Agent

Gabriel is a single-module LangChain agent exposed through FastAPI. It keeps
conversation state in PostgreSQL and exposes both JSON and Server-Sent Events
chat endpoints.

The service intentionally has no authentication in v1 and binds to loopback by
default. Do not expose it directly to the public internet.

## Requirements

- Python 3.14
- `uv`
- PostgreSQL with permission to create the `gabriel_db` database

## Setup

```bash
uv sync
createdb -h localhost -U postgres gabriel_db
uv run alembic upgrade head
```

Copy `.env.example` to `.env` and fill in the real database and API credentials.
The local `.env` file is ignored by Git.

Start the API:

```bash
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8086
```

OpenAPI documentation is available at <http://127.0.0.1:8086/docs> in
development.

## API

Send a JSON chat request:

```bash
curl -X POST http://127.0.0.1:8086/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"3fa85f64-5717-4562-b3fc-2c963f66afa6","message":"My name is Ada."}'
```

Send a follow-up using the same `session_id` to use persisted conversation
memory. Stream a response with:

```bash
curl -N -X POST http://127.0.0.1:8086/api/chat/stream \
  -H 'Content-Type: application/json' \
  -H 'Accept: text/event-stream' \
  -d '{"session_id":"3fa85f64-5717-4562-b3fc-2c963f66afa6","message":"What is my name?"}'
```

Read history:

```bash
curl 'http://127.0.0.1:8086/api/chat/sessions/3fa85f64-5717-4562-b3fc-2c963f66afa6/messages?limit=50'
```

Delete a session and its messages:

```bash
curl -X DELETE \
  http://127.0.0.1:8086/api/chat/sessions/3fa85f64-5717-4562-b3fc-2c963f66afa6
```

Health endpoints are `/health/live` and `/health/ready`.

## Configuration

Environment variables are documented in `.env.example`. `RAG_ENABLED` is
`false` by default. Embedding and vector settings are validated only when RAG is
enabled; v1 does not call embeddings or create vector tables.

## Quality Checks

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy app
uv run pytest
```

Integration tests run only when `TEST_DATABASE_URL` is set to an async SQLAlchemy
URL such as `postgresql+asyncpg://postgres@localhost/gabriel_db_test`. The live
provider test is opt-in with `RUN_PROVIDER_TESTS=true`. Apply
`uv run alembic upgrade head` to the test database before running the
repository integration test.

## Deployment

`deploy/gabriel-agent.service` runs one Uvicorn worker, which is required for
the in-process per-session lock. Production environment values belong in
`/etc/default/gabriel-agent`; an example is provided in
`deploy/gabriel-agent.env.example`.
