PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS product_events (
    event_id TEXT PRIMARY KEY,
    occurred_on TEXT NOT NULL,
    installation_id TEXT NOT NULL,
    event_name TEXT NOT NULL,
    version TEXT NOT NULL,
    platform TEXT NOT NULL,
    python TEXT NOT NULL,
    properties_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_product_events_day
    ON product_events(occurred_on);
CREATE INDEX IF NOT EXISTS idx_product_events_name_day
    ON product_events(event_name, occurred_on);
CREATE INDEX IF NOT EXISTS idx_product_events_installation
    ON product_events(installation_id);

-- Savings rows retain their monthly pseudonym only for the retention window.
-- After that, prune() folds them into the identifier-free singleton below.
CREATE TABLE IF NOT EXISTS savings_contributions (
    event_id TEXT PRIMARY KEY,
    installation_id TEXT NOT NULL,
    occurred_on TEXT NOT NULL,
    tokens_saved_thousands INTEGER NOT NULL CHECK (tokens_saved_thousands >= 0),
    modeled_cost_saved_cents INTEGER NOT NULL CHECK (modeled_cost_saved_cents >= 0)
);

CREATE INDEX IF NOT EXISTS idx_savings_contributions_day
    ON savings_contributions(occurred_on);
CREATE INDEX IF NOT EXISTS idx_savings_contributions_installation
    ON savings_contributions(installation_id);

CREATE TABLE IF NOT EXISTS savings_archive (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    tokens_saved_thousands INTEGER NOT NULL DEFAULT 0,
    modeled_cost_saved_cents INTEGER NOT NULL DEFAULT 0,
    contribution_events INTEGER NOT NULL DEFAULT 0,
    archived_through TEXT
);

INSERT OR IGNORE INTO savings_archive (
    id, tokens_saved_thousands, modeled_cost_saved_cents, contribution_events
) VALUES (1, 0, 0, 0);
