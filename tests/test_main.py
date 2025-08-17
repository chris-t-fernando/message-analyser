import json
import os
import sys
from datetime import datetime, timedelta
import runpy
import uvicorn

import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
os.environ.update({
    "DATABASE__HOST": "h",
    "DATABASE__PORT": "1",
    "DATABASE__USER": "u",
    "DATABASE__PASSWORD": "p",
    "DATABASE__NAME": "n",
})
import main


class SeqCursor:
    def __init__(self, results=None, rowcount=1):
        self.results = list(results or [])
        self.rowcount = rowcount
        self.executed = []
        self._current = None

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        if self.results:
            self._current = self.results.pop(0)
        else:
            self._current = None

    def fetchall(self):
        return self._current

    def fetchone(self):
        return self._current

    def close(self):
        pass


class DummyConn:
    def __init__(self, cursor):
        self.cursor_obj = cursor
        self.committed = False

    def cursor(self, cursor_factory=None):
        return self.cursor_obj

    def commit(self):
        self.committed = True

    def close(self):
        pass


@pytest.fixture
def client():
    return TestClient(main.app)


def test_get_conn(monkeypatch):
    called = {}
    def fake_connect(**kwargs):
        called.update(kwargs)
        return object()
    monkeypatch.setattr(main.psycopg2, "connect", fake_connect)
    main.get_conn()
    assert called["host"] == "h" and called["dbname"] == "n"


def test_parse_sender():
    assert main.parse_sender(None) == ("Hayley", None)
    assert main.parse_sender("+1 (417)017-9500") == ("Chris", "14170179500")
    assert main.parse_sender("123") == ("Hayley", "123")


def test_execute_search_groups(monkeypatch):
    cursor = SeqCursor([
        [{"id": 1}, {"id": 2}, {"id": 15}],
        [
            {"id": 1, "msg_date": 1, "sender": "A", "phone": "p", "text": "t", "tags": []},
            {"id": 2, "msg_date": 2, "sender": "A", "phone": "p", "text": "t2", "tags": []},
        ],
        [
            {"id": 15, "msg_date": 3, "sender": "B", "phone": "p", "text": "x", "tags": []}
        ],
    ])
    conn = DummyConn(cursor)
    monkeypatch.setattr(main, "get_conn", lambda: conn)
    res = main._execute_search("SQL", ["param"])
    assert res["groups"][0]["match_indices"] == [0, 1]
    assert res["groups"][1]["match_indices"] == [5]


def test_execute_search_empty(monkeypatch):
    cursor = SeqCursor([[]])
    conn = DummyConn(cursor)
    monkeypatch.setattr(main, "get_conn", lambda: conn)
    assert main._execute_search("SQL", []) == {"groups": []}


def test_index_upload_and_conversations_pages(client):
    assert "<html" in main.index()
    assert "<html" in main.upload_page()
    assert "<html" in main.conversations_page()
    assert "<html" in main.monthly_page()


def test_upload(monkeypatch, client):
    csv_content = "Date,Sender,Text\n2023-01-01,A,hi".encode("utf-8")
    cursor = SeqCursor()
    conn = DummyConn(cursor)
    monkeypatch.setattr(main, "get_conn", lambda: conn)
    response = client.post("/upload", files={"csv_file": ("f.csv", csv_content)})
    assert response.json() == {
        "lines": 1,
        "inserted": 1,
        "skipped": 0,
        "status": "insert complete",
    }


def test_search_and_tag(monkeypatch):
    called = {}

    def fake_exec(sql, params):
        called["sql"] = sql
        called["params"] = params
        return {"groups": []}

    monkeypatch.setattr(main, "_execute_search", fake_exec)
    assert main.search(["hello"], operator="OR", start="s", end="e") == {"groups": []}
    assert "LOWER(m.text) LIKE" in called["sql"]
    assert main.search_tag("tag") == {"groups": []}
    assert "LOWER(t.tag)" in called["sql"]


def test_get_messages(monkeypatch):
    rows = [{"id": 1, "Date": "d", "Sender": "s", "phone": "p", "Text": "t", "tags": []}]
    cursor = SeqCursor([rows])
    conn = DummyConn(cursor)
    monkeypatch.setattr(main, "get_conn", lambda: conn)
    res = main.get_messages(1, 1)
    assert res["rows"] == rows


def test_message_context(monkeypatch):
    rows = [
        {"id": 1, "Date": "d", "Sender": "s", "phone": "p", "Text": "t", "tags": []},
        {"id": 2, "Date": "d", "Sender": "s", "phone": "p", "Text": "t", "tags": []},
    ]
    cursor = SeqCursor([rows])
    conn = DummyConn(cursor)
    monkeypatch.setattr(main, "get_conn", lambda: conn)
    res = main.message_context(1)
    assert res["groups"][0]["match_indices"] == [0]

    cursor2 = SeqCursor([rows])
    conn2 = DummyConn(cursor2)
    monkeypatch.setattr(main, "get_conn", lambda: conn2)
    assert main.message_context(3) == {"groups": []}


def test_messages_per_month(monkeypatch):
    rows = [
        ("2023-01", "Chris", 2),
        ("2023-01", "Hayley", 1),
        ("2023-02", "Chris", 3),
    ]
    cursor = SeqCursor([rows])
    conn = DummyConn(cursor)
    monkeypatch.setattr(main, "get_conn", lambda: conn)
    res = main.messages_per_month()
    assert res == {
        "months": [
            {"month": "2023-01", "Chris": 2, "Hayley": 1},
            {"month": "2023-02", "Chris": 3, "Hayley": 0},
        ]
    }


def test_add_and_list_tags(monkeypatch):
    cursor1 = SeqCursor()
    conn1 = DummyConn(cursor1)
    cursor2 = SeqCursor([[("a",), ("b",)]])
    conn2 = DummyConn(cursor2)
    conns = [conn1, conn2]
    monkeypatch.setattr(main, "get_conn", lambda: conns.pop(0))
    assert main.add_tag(1, tag="x") == {"status": "ok"}
    assert main.list_tags() == {"tags": ["a", "b"]}


def test_get_wordcloud(monkeypatch):
    cursor = SeqCursor([None, ({"a": 1, "the": 2, "b1": 3},)])
    conn = DummyConn(cursor)
    monkeypatch.setattr(main, "get_conn", lambda: conn)
    res = main.get_wordcloud()
    assert res == {"words": {}}

    cursor2 = SeqCursor([None, None])
    conn2 = DummyConn(cursor2)
    monkeypatch.setattr(main, "get_conn", lambda: conn2)
    assert main.get_wordcloud() == {"words": None}


def test_generate_wordcloud(monkeypatch):
    cursor = SeqCursor([None, []])
    conn = DummyConn(cursor)
    monkeypatch.setattr(main, "get_conn", lambda: conn)
    assert main.generate_wordcloud() == {"status": "no_messages"}

    cursor2 = SeqCursor([None, [("hello world",)], None])
    conn2 = DummyConn(cursor2)
    monkeypatch.setattr(main, "get_conn", lambda: conn2)
    res = main.generate_wordcloud()
    assert res == {"status": "generated"}
    assert conn2.committed


def test_load_conversations_and_tone(monkeypatch):
    now = datetime.now()
    rows = [
        {"msg_date": now, "sender": "Chris", "text": "good"},
        {"msg_date": now + timedelta(hours=3), "sender": "Hayley", "text": "bad"},
    ]
    cursor = SeqCursor([rows])
    conn = DummyConn(cursor)
    monkeypatch.setattr(main, "get_conn", lambda: conn)
    convs = main._load_conversations()
    assert len(convs) == 2
    tone = main._tone([{"sender": "Chris", "text": "great"}, {"sender": "Hayley", "text": "awful"}])
    assert tone["Chris"] == "positive" and tone["Hayley"] == "negative"


def test_tone_neutral():
    tone = main._tone([{"sender": "Chris", "text": ""}])
    assert tone["Hayley"] == "neutral"


def test_conversations_api_and_detail(monkeypatch):
    convs = [
        [
            {"msg_date": datetime(2023, 1, 1), "sender": "Chris", "text": "hi"},
            {"msg_date": datetime(2023, 1, 1, 0, 1), "sender": "Hayley", "text": "yo"},
        ]
    ]
    monkeypatch.setattr(main, "_load_conversations", lambda: convs)
    api = main.conversations_api()
    assert api["conversations"][0]["count"] == 2
    assert main.conversation_detail(0)["messages"] == convs[0]
    assert main.conversation_detail(5) == {"messages": []}


def test_main_entry(monkeypatch):
    called = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, host, port, reload: called.setdefault("args", (app, host, port, reload)))
    runpy.run_module("main", run_name="__main__")
    assert called["args"] == ("main:app", "0.0.0.0", 8000, True)
