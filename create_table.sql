CREATE TABLE IF NOT EXISTS messages (
    id SERIAL PRIMARY KEY,
    msg_date TEXT,
    sender TEXT,
    received TEXT,
    imessage TEXT,
    text TEXT,
    UNIQUE (msg_date, sender, text)
);
