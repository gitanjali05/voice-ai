import json
import os
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from data_tools import df_overview, group_mean, load_csv, missing_values, top_rows, value_counts

load_dotenv()

MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
SYSTEM_PROMPT = (
    "You are a data analyst assistant for CSV files in the local data folder. "
    "Ask for the CSV filename when missing. Keep responses concise. "
    "Use tools for factual CSV operations; do not invent data."
)

client = OpenAI()


# In-memory session state for lightweight demo use.
SESSIONS: dict[str, list[dict[str, Any]]] = {}
DF_CACHE: dict[str, Any] = {}


def _normalize_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _normalize_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _get_df(filename: str):
    key = os.path.basename(filename)
    if key not in DF_CACHE:
        DF_CACHE[key] = load_csv(key)
    return DF_CACHE[key]


def tool_csv_overview(filename: str) -> dict[str, Any]:
    df = _get_df(filename)
    info = df_overview(df)
    return {
        "rows": info["rows"],
        "cols": info["cols"],
        "columns": info["columns"],
        "message": f"{filename} has {info['rows']} rows and {info['cols']} columns.",
    }


def tool_csv_head(filename: str, n: int = 5) -> dict[str, Any]:
    df = _get_df(filename)
    safe_n = max(1, min(int(n), 20))
    rows = top_rows(df, safe_n)
    return {
        "rows": rows,
        "message": f"Showing first {len(rows)} rows from {filename}.",
    }


def tool_csv_missing(filename: str, top_n: int = 10) -> dict[str, Any]:
    df = _get_df(filename)
    safe_n = max(1, min(int(top_n), 50))
    mv = missing_values(df, top_n=safe_n)
    if not mv["missing"]:
        return {"message": f"No missing values found in {filename}."}
    return {
        "missing": mv["missing"],
        "message": f"Top missing-value columns in {filename}.",
    }


def tool_csv_value_counts(filename: str, column: str, n: int = 10) -> dict[str, Any]:
    df = _get_df(filename)
    safe_n = max(1, min(int(n), 50))
    counts = value_counts(df, column, n=safe_n)
    return {
        "counts": counts["counts"],
        "message": f"Top values in {column} from {filename}.",
    }


def tool_csv_group_mean(filename: str, group_col: str, metric_col: str, n: int = 10) -> dict[str, Any]:
    df = _get_df(filename)
    safe_n = max(1, min(int(n), 50))
    out = group_mean(df, group_col, metric_col, n=safe_n)
    return {
        "mean_by_group": out["mean_by_group"],
        "message": f"Mean {metric_col} by {group_col}.",
    }


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "csv_overview",
            "description": "Show number of rows, columns, and column names for a CSV file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "CSV filename in data/"}
                },
                "required": ["filename"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "csv_head",
            "description": "Return first N rows of a CSV.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "n": {"type": "integer", "default": 5, "minimum": 1, "maximum": 20},
                },
                "required": ["filename"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "csv_missing",
            "description": "List columns with missing values.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "top_n": {"type": "integer", "default": 10, "minimum": 1, "maximum": 50},
                },
                "required": ["filename"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "csv_value_counts",
            "description": "Get top value counts for a column.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "column": {"type": "string"},
                    "n": {"type": "integer", "default": 10, "minimum": 1, "maximum": 50},
                },
                "required": ["filename", "column"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "csv_group_mean",
            "description": "Compute mean of metric column grouped by another column.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "group_col": {"type": "string"},
                    "metric_col": {"type": "string"},
                    "n": {"type": "integer", "default": 10, "minimum": 1, "maximum": 50},
                },
                "required": ["filename", "group_col", "metric_col"],
                "additionalProperties": False,
            },
        },
    },
]

TOOL_IMPL = {
    "csv_overview": tool_csv_overview,
    "csv_head": tool_csv_head,
    "csv_missing": tool_csv_missing,
    "csv_value_counts": tool_csv_value_counts,
    "csv_group_mean": tool_csv_group_mean,
}


def _tool_error_message(exc: Exception) -> str:
    if isinstance(exc, FileNotFoundError):
        return str(exc)
    if isinstance(exc, ValueError):
        return str(exc)
    if isinstance(exc, TypeError):
        return str(exc)
    return "Tool failed unexpectedly. Please check file and column names."


def run_agent_turn(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    work_messages = list(messages)

    while True:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + work_messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.2,
        )

        msg = completion.choices[0].message
        assistant_message: dict[str, Any] = {"role": "assistant"}
        if msg.content is not None:
            assistant_message["content"] = msg.content
        if msg.tool_calls:
            assistant_message["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]

        work_messages.append(assistant_message)

        if not msg.tool_calls:
            return (msg.content or "I couldn't produce a response."), work_messages

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            raw_args = tc.function.arguments or "{}"

            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}

            try:
                result = TOOL_IMPL[fn_name](**args)
                payload = _normalize_jsonable({"ok": True, "result": result})
            except Exception as exc:
                payload = {"ok": False, "error": _tool_error_message(exc)}

            work_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(payload),
                }
            )


HTML = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>CSV Voice Agent - Web Chat</title>
  <style>
    :root {
      --bg: #f7f9f4;
      --panel: #ffffff;
      --text: #1e2a22;
      --muted: #607066;
      --brand: #1f6f5c;
      --brand-2: #cce3da;
      --border: #dce6e0;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background: radial-gradient(circle at 15% 15%, #e8f3ed, transparent 45%), var(--bg);
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 24px;
    }
    .app {
      width: min(860px, 100%);
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      box-shadow: 0 10px 30px rgba(20, 46, 34, 0.08);
      overflow: hidden;
    }
    .head {
      background: linear-gradient(90deg, var(--brand), #2f8c73);
      color: white;
      padding: 14px 16px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
    }
    .head h1 {
      margin: 0;
      font-size: 16px;
      font-weight: 700;
    }
    .head small { opacity: 0.9; }
    #chat {
      height: min(62vh, 680px);
      overflow: auto;
      padding: 14px;
      background: linear-gradient(180deg, #fafcf9, #f6faf8);
    }
    .msg {
      margin: 10px 0;
      padding: 10px 12px;
      border-radius: 10px;
      max-width: 85%;
      white-space: pre-wrap;
      line-height: 1.4;
      border: 1px solid var(--border);
    }
    .user { margin-left: auto; background: var(--brand-2); }
    .assistant { background: #fff; }
    .meta { font-size: 12px; color: var(--muted); margin-bottom: 4px; }
    form {
      border-top: 1px solid var(--border);
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      padding: 12px;
      background: #fff;
    }
    input {
      width: 100%;
      font: inherit;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 11px 12px;
      outline: none;
    }
    input:focus { border-color: var(--brand); }
    button {
      border: 0;
      border-radius: 10px;
      background: var(--brand);
      color: white;
      font: inherit;
      padding: 10px 14px;
      cursor: pointer;
    }
    button:disabled { opacity: 0.6; cursor: default; }
  </style>
</head>
<body>
  <div class=\"app\">
    <div class=\"head\">
      <h1>CSV Data Agent (Web Chat)</h1>
      <small>Put files inside <code>data/</code></small>
    </div>
    <div id=\"chat\"></div>
    <form id=\"form\">
      <input id=\"input\" placeholder=\"Example: overview of sales.csv\" autocomplete=\"off\" />
      <button id=\"send\" type=\"submit\">Send</button>
    </form>
  </div>

<script>
  const chat = document.getElementById('chat');
  const form = document.getElementById('form');
  const input = document.getElementById('input');
  const send = document.getElementById('send');

  function addMsg(role, text) {
    const wrap = document.createElement('div');
    wrap.className = `msg ${role}`;
    const meta = document.createElement('div');
    meta.className = 'meta';
    meta.textContent = role === 'user' ? 'You' : 'Agent';
    const body = document.createElement('div');
    body.textContent = text;
    wrap.append(meta, body);
    chat.appendChild(wrap);
    chat.scrollTop = chat.scrollHeight;
  }

  addMsg('assistant', 'Ready. Ask about any CSV in the data folder, and include the filename when possible.');

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;

    addMsg('user', text);
    input.value = '';
    send.disabled = true;

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({message: text})
      });
      const data = await res.json();
      if (!res.ok) {
        addMsg('assistant', data.error || 'Request failed.');
      } else {
        addMsg('assistant', data.reply || '(No reply)');
      }
    } catch (err) {
      addMsg('assistant', 'Network error. Is server running?');
    } finally {
      send.disabled = false;
      input.focus();
    }
  });
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def _session_id(self) -> str:
        cookie = self.headers.get("Cookie", "")
        for chunk in cookie.split(";"):
            part = chunk.strip()
            if part.startswith("sid="):
                return part[4:]
        return ""

    def _ensure_session(self) -> tuple[str, bool]:
        sid = self._session_id()
        is_new = False
        if not sid:
            sid = uuid.uuid4().hex
            is_new = True
        if sid not in SESSIONS:
            SESSIONS[sid] = []
        return sid, is_new

    def _send_json(self, status: int, payload: dict[str, Any], set_sid: str | None = None) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        if set_sid:
            self.send_header("Set-Cookie", f"sid={set_sid}; Path=/; HttpOnly; SameSite=Lax")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            body = HTML.encode("utf-8")
            sid, is_new = self._ensure_session()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            if is_new:
                self.send_header("Set-Cookie", f"sid={sid}; Path=/; HttpOnly; SameSite=Lax")
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def do_POST(self) -> None:
        if self.path != "/api/chat":
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
            return

        sid, is_new = self._ensure_session()

        try:
            content_len = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_len)
            req = json.loads(raw or b"{}")
            user_message = str(req.get("message", "")).strip()
            if not user_message:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "message is required"}, sid if is_new else None)
                return

            history = SESSIONS[sid]
            history.append({"role": "user", "content": user_message})

            reply, updated = run_agent_turn(history)
            SESSIONS[sid] = updated[-24:]

            self._send_json(HTTPStatus.OK, {"reply": reply}, sid if is_new else None)
        except Exception as exc:
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"Server error: {exc}"},
                sid if is_new else None,
            )


def main() -> None:
    host = os.getenv("WEB_CHAT_HOST", "127.0.0.1")
    port = int(os.getenv("WEB_CHAT_PORT", "8000"))
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required in .env")

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Web chat running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
