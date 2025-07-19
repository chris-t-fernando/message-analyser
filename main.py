import json
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import csv
import io
import re
from collections import Counter
import psycopg2
from psycopg2.extras import RealDictCursor
from config import get_settings

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



def get_conn():
    cfg = settings.database
    return psycopg2.connect(
        host=cfg.host,
        port=cfg.port,
        user=cfg.user,
        password=cfg.password,
        dbname=cfg.name,
    )


@app.get("/", response_class=HTMLResponse)
def index():
    return Path("static/index.html").read_text(encoding="utf-8")


@app.get("/upload", response_class=HTMLResponse)
def upload_page():
    return Path("static/upload.html").read_text(encoding="utf-8")


@app.post("/upload")
async def upload(csv_file: UploadFile = File(...)):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id SERIAL PRIMARY KEY,
            msg_date TEXT,
            sender TEXT,
            received TEXT,
            imessage TEXT,
            text TEXT,
            UNIQUE(msg_date, sender, text)
        )
        """
    )

    content = await csv_file.read()
    reader = csv.DictReader(io.StringIO(content.decode("utf-8")))
    inserted = 0
    for row in reader:
        cur.execute(
            """
            INSERT INTO messages (msg_date, sender, received, imessage, text)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
            """,
            (
                row.get("Date"),
                row.get("Sender"),
                row.get("Received"),
                row.get("iMessage"),
                row.get("Text"),
            ),
        )
        if cur.rowcount > 0:
            inserted += 1

    conn.commit()
    cur.close()
    conn.close()
    return {"inserted": inserted}


@app.get("/search")
def search(query: str = Query(..., min_length=1)):
    """Search for all rows containing the query and return groups with context."""
    conn = get_conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    # Find all matching ids ordered by id
    cur.execute(
        """
        SELECT id
        FROM messages
        WHERE LOWER(text) LIKE %s
        ORDER BY id
        """,
        (f"%{query.lower()}%",),
    )
    id_rows = cur.fetchall()
    if not id_rows:
        cur.close()
        conn.close()
        return {"groups": []}

    ids = [r["id"] for r in id_rows]

    # Group ids that fall within 10 lines of each other (5 before/after)
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
            SELECT id, msg_date AS "Date", sender AS "Sender", received AS "Received", imessage AS "iMessage", text AS "Text"
            FROM messages
            WHERE id BETWEEN %s AND %s
            ORDER BY id
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
        return {"words": row[0]}
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
    words = re.findall(r"\w+", text.lower())
    counts = Counter(words)
    cur.execute("INSERT INTO wordclouds (data) VALUES (%s)", (json.dumps(counts),))
    conn.commit()
    cur.close()
    conn.close()
    return {"status": "generated"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
