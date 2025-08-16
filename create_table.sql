CREATE TABLE IF NOT EXISTS messages (
    id SERIAL PRIMARY KEY,
    msg_date TEXT,
    sender TEXT,
    phone TEXT,
    text TEXT,
    UNIQUE (msg_date, sender, text, phone)
);

CREATE TABLE IF NOT EXISTS wordclouds (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    data JSONB
);

CREATE TABLE IF NOT EXISTS tags (
    id SERIAL PRIMARY KEY,
    message_id INTEGER REFERENCES messages(id) ON DELETE CASCADE,
    tag TEXT NOT NULL,
    UNIQUE (message_id, tag)
);

CREATE INDEX IF NOT EXISTS idx_tags_lower_tag ON tags (LOWER(tag));
