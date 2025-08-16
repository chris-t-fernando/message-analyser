CREATE TABLE IF NOT EXISTS tags (
    id SERIAL PRIMARY KEY,
    message_id INTEGER REFERENCES messages(id) ON DELETE CASCADE,
    tag TEXT NOT NULL,
    UNIQUE (message_id, tag)
);

CREATE INDEX IF NOT EXISTS idx_tags_lower_tag ON tags (LOWER(tag));
