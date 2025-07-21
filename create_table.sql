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
