-- Review flags raised on atlas steps (Cloudflare D1). Run once:
--   npx wrangler d1 execute atlas-review --remote --file=review/schema.sql
CREATE TABLE IF NOT EXISTS flags (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created TEXT NOT NULL,
  proc TEXT NOT NULL,          -- approach key, e.g. mvr-std
  approach TEXT,               -- operation and approach as shown
  step TEXT,                   -- step id
  step_title TEXT,
  kind TEXT,                   -- wrong | outdated | unclear | missing | typo | other
  comment TEXT NOT NULL,
  name TEXT,
  status TEXT NOT NULL DEFAULT 'open',   -- open | fixed | rejected
  resolution TEXT,
  resolved TEXT
);
CREATE INDEX IF NOT EXISTS flags_status ON flags(status);
