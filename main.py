from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import csv
from pathlib import Path

app = FastAPI()
# allow requests from the demo page
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

csv_path = Path(__file__).parent / "sample.csv"
rows = []
with open(csv_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    return Path("static/index.html").read_text(encoding="utf-8")


@app.get("/search")
def search(query: str = Query(..., min_length=1)):
    query_lower = query.lower()
    for idx, row in enumerate(rows):
        if query_lower in row["Text"].lower():
            start = max(idx - 5, 0)
            end = min(idx + 5, len(rows) - 1)
            subset = rows[start : end + 1]
            return {"found_index": idx, "start": start, "end": end, "rows": subset}
    return {"found_index": None, "start": None, "end": None, "rows": []}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
