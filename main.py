import json
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import csv
import io
import os
import base64
from io import BytesIO
from wordcloud import WordCloud
import boto3
import psycopg2
from psycopg2.extras import RealDictCursor

app = FastAPI()
# allow requests from the demo page
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/static", StaticFiles(directory="static"), name="static")


def get_db_config(env: str) -> dict:
    """Load database configuration from AWS SSM."""
    ssm = boto3.client("ssm")
    keys = ["PGHOST", "PGPORT", "PGUSER", "PGPASSWORD", "PGDATABASE"]
    cfg = {}
    for key in keys:
        name = f"/stockapp/{env}/{key}"
        resp = ssm.get_parameter(Name=name, WithDecryption=True)
        value = resp["Parameter"]["Value"]
        try:
            cfg[key] = json.loads(value)
        except json.JSONDecodeError:
            cfg[key] = value
    return cfg


def get_conn(env: str):
    cfg = get_db_config(env)
    return psycopg2.connect(
        host=cfg["PGHOST"],
        port=cfg["PGPORT"],
        user=cfg["PGUSER"],
        password=cfg["PGPASSWORD"],
        dbname=cfg["PGDATABASE"],
    )


@app.get("/", response_class=HTMLResponse)
def index():
    return Path("static/index.html").read_text(encoding="utf-8")


@app.get("/upload", response_class=HTMLResponse)
def upload_page():
    return Path("static/upload.html").read_text(encoding="utf-8")


@app.post("/upload")
async def upload(csv_file: UploadFile = File(...), env: str = "devtest"):
    conn = get_conn(env)
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
def search(query: str = Query(..., min_length=1), env: str = "dev"):
    conn = get_conn(env)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute(
        """
        SELECT id, msg_date AS "Date", sender AS "Sender", received AS "Received", imessage AS "iMessage", text AS "Text"
        FROM messages
        WHERE LOWER(text) LIKE %s
        ORDER BY id
        LIMIT 1
        """,
        (f"%{query.lower()}%",),
    )
    row = cur.fetchone()
    if not row:
        cur.close()
        conn.close()
        return {"found_index": None, "start": None, "end": None, "rows": []}

    found_id = row["id"]
    start_id = max(found_id - 5, 1)
    end_id = found_id + 5
    cur.execute(
        """
        SELECT msg_date AS "Date", sender AS "Sender", received AS "Received", imessage AS "iMessage", text AS "Text"
        FROM messages
        WHERE id BETWEEN %s AND %s
        ORDER BY id
        """,
        (start_id, end_id),
    )
    subset = cur.fetchall()
    cur.close()
    conn.close()
    start_index = start_id - 1
    return {
        "found_index": found_id - 1,
        "start": start_index,
        "end": start_index + len(subset) - 1,
        "rows": subset,
    }


@app.get("/wordcloud")
def get_wordcloud(env: str = "dev"):
    conn = get_conn(env)
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE IF NOT EXISTS wordclouds (id SERIAL PRIMARY KEY, created_at TIMESTAMPTZ DEFAULT NOW(), image TEXT)"
    )
    cur.execute("SELECT image FROM wordclouds ORDER BY created_at DESC LIMIT 1")
    row = cur.fetchone()
    cur.close()
    conn.close()
    if row:
        return {"image": row[0]}
    return {"image": None}


@app.post("/generate_wordcloud")
def generate_wordcloud(env: str = "dev"):
    conn = get_conn(env)
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE IF NOT EXISTS wordclouds (id SERIAL PRIMARY KEY, created_at TIMESTAMPTZ DEFAULT NOW(), image TEXT)"
    )
    cur.execute("SELECT text FROM messages")
    rows = cur.fetchall()
    text = " ".join(r[0] for r in rows if r[0])
    if not text:
        cur.close()
        conn.close()
        return {"status": "no_messages"}
    wc = WordCloud(width=400, height=300)
    wc.generate(text)
    img = wc.to_image()
    buf = BytesIO()
    img.save(buf, format="PNG")
    img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    cur.execute("INSERT INTO wordclouds (image) VALUES (%s)", (img_str,))
    conn.commit()
    cur.close()
    conn.close()
    return {"status": "generated"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
