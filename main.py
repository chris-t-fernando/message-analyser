import json
import asyncio
from fastapi import Body, FastAPI, File, Query, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import csv
import io
import re
from collections import Counter
from datetime import timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import psycopg2
from psycopg2.extras import RealDictCursor
from config import get_settings
import nltk
from nltk.corpus import stopwords, brown

app = FastAPI()
# load configuration from environment
settings = get_settings()
# allow requests from the demo page
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/static", StaticFiles(directory="static"), name="static")

analyzer = SentimentIntensityAnalyzer()
GAP = timedelta(hours=2)

# Track progress of CSV uploads for server-sent events
UPLOAD_PROGRESS: dict[str, int | bool] | None = None

# Load English stop words using NLTK. Download the corpus if it is
# missing so the application can run in a fresh environment.
try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:  # pragma: no cover - download path tested implicitly
    nltk.download("stopwords", quiet=True)
    STOP_WORDS = set(stopwords.words("english"))

# Augment stop words with the most common terms from the Brown corpus.
try:
    common_words = nltk.FreqDist(w.lower() for w in brown.words())
except LookupError:  # pragma: no cover - download path tested implicitly
    nltk.download("brown", quiet=True)
    common_words = nltk.FreqDist(w.lower() for w in brown.words())
STOP_WORDS.update(w for w, _ in common_words.most_common(800) if w.isalpha())

# Ensure user-requested terms are excluded even if not part of the
# default corpus.
STOP_WORDS.update({"also", "please", "hi", "bit", "okay"})


def get_conn():
    cfg = settings.database
    return psycopg2.connect(
        host=cfg.host,
        port=cfg.port,
        user=cfg.user,
        password=cfg.password,
        dbname=cfg.name,
    )


def parse_sender(sender: str) -> tuple[str, str | None]:
    """Return (name, phone_digits) based on sender string."""
    if sender is None:
        return "Hayley", None
    digits = re.sub(r"\D", "", sender)
    name = "Chris" if "417017950" in digits else "Hayley"
    return name, digits or None


def _execute_search(sql: str, params: list[str]):
    """Run a search query and return grouped results with surrounding rows."""
    conn = get_conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute(sql, params)
    id_rows = cur.fetchall()
    if not id_rows:
        cur.close()
        conn.close()
        return {"groups": []}

    ids = [r["id"] for r in id_rows]
    groups: list[list[int]] = []
    current_group = [ids[0]]
    for i in ids[1:]:
        if i - current_group[-1] <= 10:
            current_group.append(i)
        else:
            groups.append(current_group)
            current_group = [i]
    groups.append(current_group)

    results: list[dict] = []
    for g in groups:
        start_id = max(g[0] - 5, 1)
        end_id = g[-1] + 5
        cur.execute(
            """
            SELECT m.id,
                   m.msg_date AS "Date",
                   m.sender AS "Sender",
                   m.phone,
                   m.text AS "Text",
                   COALESCE(array_agg(t.tag) FILTER (WHERE t.tag IS NOT NULL), ARRAY[]::text[]) AS tags
            FROM messages m
            LEFT JOIN tags t ON m.id = t.message_id
            WHERE m.id BETWEEN %s AND %s
            GROUP BY m.id
            ORDER BY m.id
            """,
            (start_id, end_id),
        )
        subset = cur.fetchall()
        match_indices = [mid - start_id for mid in g]
        results.append(
            {
                "start": start_id - 1,
                "end": start_id - 1 + len(subset) - 1,
                "match_indices": match_indices,
                "rows": subset,
            }
        )

    cur.close()
    conn.close()
    return {"groups": results}


@app.get("/", response_class=HTMLResponse)
def index():
    return Path("static/index.html").read_text(encoding="utf-8")


@app.get("/upload", response_class=HTMLResponse)
def upload_page():
    return Path("static/upload.html").read_text(encoding="utf-8")


@app.get("/upload/progress")
async def upload_progress():
    """Stream the number of rows inserted during the current upload."""

    async def event_generator():
        last_reported = -1
        while True:
            if UPLOAD_PROGRESS is None:
                await asyncio.sleep(0.5)
                continue
            current = UPLOAD_PROGRESS.get("inserted", 0)
            if current != last_reported:
                yield f"data: {current}\n\n"
                last_reported = current
            if UPLOAD_PROGRESS.get("done") and current == last_reported:
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/upload")
async def upload(csv_file: UploadFile = File(...)):
    global UPLOAD_PROGRESS
    UPLOAD_PROGRESS = {"inserted": 0, "done": False}

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id SERIAL PRIMARY KEY,
            msg_date TIMESTAMP,
            sender TEXT,
            phone TEXT,
            text TEXT,
            UNIQUE(msg_date, sender, text, phone)
        )
        """
    )

    content = await csv_file.read()
    text = content.decode("utf-8", errors="replace")
    # Normalize line endings and remove control characters that may confuse the
    # CSV parser. Some exports include lone carriage returns (\r) or other
    # unicode separators which cause the csv module to misinterpret newlines.
    reader = csv.DictReader(io.StringIO(text, newline=""))
    inserted = 0
    for row in reader:
        name, phone = parse_sender(row.get("Sender"))
        cur.execute(
            """
            INSERT INTO messages (msg_date, sender, phone, text)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT DO NOTHING
            """,
            (
                row.get("Date"),
                name,
                phone,
                row.get("Text"),
            ),
        )
        if cur.rowcount > 0:
            inserted += 1
            if inserted % 100 == 0:
                UPLOAD_PROGRESS["inserted"] = inserted

    conn.commit()
    cur.close()
    conn.close()
    UPLOAD_PROGRESS.update({"inserted": inserted, "done": True})
    return {"inserted": inserted}


@app.get("/search")
def search(
    terms: list[str] = Query(...),
    operator: str = Query("AND"),
    start: str | None = None,
    end: str | None = None,
):
    """Search for rows containing terms with optional date filtering."""
    op = "AND" if operator.upper() != "OR" else "OR"
    term_clauses: list[str] = []
    params: list[str] = []
    for t in terms:
        pattern = f"%{t.lower()}%"
        term_clauses.append(
            "(LOWER(m.text) LIKE %s OR EXISTS (SELECT 1 FROM tags t WHERE t.message_id = m.id AND LOWER(t.tag) LIKE %s))"
        )
        params.extend([pattern, pattern])
    clause_sql = f" {op} ".join(term_clauses)
    where_parts = [f"({clause_sql})"]
    if start:
        where_parts.append("msg_date >= %s")
        params.append(start)
    if end:
        where_parts.append("msg_date <= %s")
        params.append(end)
    where_sql = " AND ".join(where_parts)
    sql = f"SELECT m.id FROM messages m WHERE {where_sql} ORDER BY m.id"
    return _execute_search(sql, params)


@app.get("/search_tag")
def search_tag(tag: str):
    """Return rows tagged with ``tag`` using the same grouping logic."""
    sql = (
        """
        SELECT m.id
        FROM messages m
        JOIN tags t ON m.id = t.message_id
        WHERE LOWER(t.tag) = %s
        ORDER BY m.id
        """
    )
    return _execute_search(sql, [tag.lower()])


@app.get("/messages")
def get_messages(start: int, count: int = 5):
    """Return a slice of messages starting at ``start`` for ``count`` rows."""
    conn = get_conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute(
        """
        SELECT m.id,
               m.msg_date AS "Date",
               m.sender AS "Sender",
               m.phone,
               m.text AS "Text",
               COALESCE(array_agg(t.tag) FILTER (WHERE t.tag IS NOT NULL), ARRAY[]::text[]) AS tags
        FROM messages m
        LEFT JOIN tags t ON m.id = t.message_id
        WHERE m.id >= %s AND m.id < %s
        GROUP BY m.id
        ORDER BY m.id
        """,
        (start, start + count),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return {"rows": rows}


@app.get("/message/{mid}")
def message_context(mid: int):
    """Return ``mid`` with surrounding messages formatted like a search result."""
    start_id = max(mid - 5, 1)
    end_id = mid + 5
    conn = get_conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute(
        """
        SELECT m.id,
               m.msg_date AS "Date",
               m.sender AS "Sender",
               m.phone,
               m.text AS "Text",
               COALESCE(array_agg(t.tag) FILTER (WHERE t.tag IS NOT NULL), ARRAY[]::text[]) AS tags
        FROM messages m
        LEFT JOIN tags t ON m.id = t.message_id
        WHERE m.id BETWEEN %s AND %s
        GROUP BY m.id
        ORDER BY m.id
        """,
        (start_id, end_id),
    )
    subset = cur.fetchall()
    cur.close()
    conn.close()
    try:
        match_index = next(i for i, r in enumerate(subset) if r["id"] == mid)
    except StopIteration:
        return {"groups": []}
    return {
        "groups": [
            {
                "start": start_id - 1,
                "end": start_id - 1 + len(subset) - 1,
                "match_indices": [match_index],
                "rows": subset,
            }
        ]
    }


@app.post("/messages/{mid}/tags")
def add_tag(mid: int, tag: str = Body(..., embed=True)):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS tags (
            id SERIAL PRIMARY KEY,
            message_id INTEGER REFERENCES messages(id) ON DELETE CASCADE,
            tag TEXT NOT NULL,
            UNIQUE (message_id, tag)
        )
        """
    )
    cur.execute(
        "INSERT INTO tags (message_id, tag) VALUES (%s, %s) ON CONFLICT DO NOTHING",
        (mid, tag),
    )
    conn.commit()
    cur.close()
    conn.close()
    return {"status": "ok"}


@app.get("/tags")
def list_tags():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT tag FROM tags ORDER BY tag")
    tags = [r[0] for r in cur.fetchall()]
    cur.close()
    conn.close()
    return {"tags": tags}


@app.get("/wordcloud")
def get_wordcloud():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE IF NOT EXISTS wordclouds (id SERIAL PRIMARY KEY, created_at TIMESTAMPTZ DEFAULT NOW(), data JSONB)"
    )
    cur.execute("SELECT data FROM wordclouds ORDER BY created_at DESC LIMIT 1")
    row = cur.fetchone()
    cur.close()
    conn.close()
    if row:
        data = row[0]
        if isinstance(data, dict):
            # Filter out stop words from stored data in case previous
            # generations included them.
            filtered = {
                k: v
                for k, v in data.items()
                if k not in STOP_WORDS and not any(c.isdigit() for c in k)
            }
            items = sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:200]
            data = {k: v for k, v in items}
        return {"words": data}
    return {"words": None}


@app.post("/generate_wordcloud")
def generate_wordcloud():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE IF NOT EXISTS wordclouds (id SERIAL PRIMARY KEY, created_at TIMESTAMPTZ DEFAULT NOW(), data JSONB)"
    )
    cur.execute("SELECT text FROM messages")
    rows = cur.fetchall()
    text = " ".join(r[0] for r in rows if r[0])
    if not text:
        cur.close()
        conn.close()
        return {"status": "no_messages"}
    words = re.findall(r"[a-z']+", text.lower())
    words = [w for w in words if w not in STOP_WORDS]
    counts = Counter(words)
    cur.execute("INSERT INTO wordclouds (data) VALUES (%s)", (json.dumps(counts),))
    conn.commit()
    cur.close()
    conn.close()
    return {"status": "generated"}


@app.get("/api/messages_per_month")
def messages_per_month():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT to_char(msg_date, 'YYYY-MM') AS month, sender, COUNT(*)
        FROM messages
        GROUP BY 1, 2
        ORDER BY 1
        """
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    data: dict[str, dict[str, int]] = {}
    for month, sender, count in rows:
        if month not in data:
            data[month] = {"Chris": 0, "Hayley": 0}
        data[month][sender] = count
    result = [
        {"month": m, "Chris": v["Chris"], "Hayley": v["Hayley"]}
        for m, v in sorted(data.items())
    ]
    return {"months": result}


@app.get("/monthly", response_class=HTMLResponse)
def monthly_page():
    return Path("static/monthly.html").read_text(encoding="utf-8")


def _load_conversations():
    conn = get_conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute(
        "SELECT msg_date, sender, text FROM messages ORDER BY msg_date"
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    conversations: list[list[dict]] = []
    current: list[dict] = []
    last_time = None
    for row in rows:
        msg_time = row["msg_date"]
        if last_time and msg_time - last_time > GAP:
            conversations.append(current)
            current = []
        current.append(row)
        last_time = msg_time
    if current:
        conversations.append(current)
    return conversations


def _tone(messages: list[dict]) -> dict:
    scores = {"Chris": [], "Hayley": []}
    for m in messages:
        score = analyzer.polarity_scores(m["text"] or "")['compound']
        scores[m["sender"]].append(score)
    tone = {}
    for name, vals in scores.items():
        if vals:
            avg = sum(vals) / len(vals)
            if avg > 0.05:
                tone[name] = "positive"
            elif avg < -0.05:
                tone[name] = "negative"
            else:
                tone[name] = "neutral"
        else:
            tone[name] = "neutral"
    return tone


@app.get("/api/conversations")
def conversations_api():
    conversations = _load_conversations()
    summaries = []
    for idx, conv in enumerate(conversations):
        summaries.append(
            {
                "id": idx,
                "start": conv[0]["msg_date"],
                "end": conv[-1]["msg_date"],
                "tone": _tone(conv),
                "count": len(conv),
            }
        )
    return {"conversations": summaries}


@app.get("/api/conversations/{cid}")
def conversation_detail(cid: int):
    conversations = _load_conversations()
    if cid < 0 or cid >= len(conversations):
        return {"messages": []}
    return {"messages": conversations[cid]}


@app.get("/conversations", response_class=HTMLResponse)
def conversations_page():
    return Path("static/conversations.html").read_text(encoding="utf-8")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
