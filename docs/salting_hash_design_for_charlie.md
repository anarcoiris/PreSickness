# Randomized Salting Hash Design

**Target Agent:** Charlie Bot / Charlie API
**Context:** MS-Predictor Unified API Security Update

## Problem Statement
Currently, patient/user identifiers (`user_id_hash`) are generated deterministically using an unsalted SHA-256 hash of their email address:
`user_id_hash = hashlib.sha256(email.encode()).hexdigest()[:16]`

While this obscures the email superficially, it's vulnerable to rainbow table attacks and dictionary attacks. Given that this is a healthcare application handling "PreSickness" data, PII must be protected with cryptography standard designs.

## Proposed Design: Randomized Salt + HMAC-SHA256 (or bcrypt)

To ensure true pseudo-anonymization and protection against offline reversing, the ID generation should employ a randomly generated per-user salt.

### Step-by-Step Implementation for Charlie

1. **Database Schema Update**
   - Add a new column `user_salt` (VARCHAR) to the `users` table.
   - Alternatively, transition to standard UUIDv4 as the primary key and drop email-based hashing entirely. (Note: A transition to `UUIDv4` was just implemented in the API for new users, removing the hash derivation entirely, which is optimal).

2. **If keeping Email-derived Hashes (with Salt)**
   - When a user registers, generate a cryptographic salt: `salt = secrets.token_hex(16)`
   - Create the hash: `user_id_hash = hmac.new(salt.encode(), email.encode(), hashlib.sha256).hexdigest()[:16]`
   - Store both `user_id_hash` and `salt` in the database.
   - **Important:** If `user_id_hash` is used to index data everywhere (TimescaleDB hypertable for metrics, uploads, events), the `user_id_hash` MUST be permanently stored and never re-derived without the original salt.

3. **Transition to Pure UUID (Recommended & Currently Implemented)**
   - **Generation:** `user_id_hash = uuid.uuid4().hex[:16]`
   - **Why:** Detaching the user ID completely from the email means total untraceability from the primary key. If a database dump is leaked, the IDs cannot be mathematically linked back to the `users` table's email column without the relational pairing.
   - **Charlie's Task:** Build a migration script that:
     1. Creates a new random UUID for all existing users in the `users` table.
     2. Cascades this update to `datapoints`, `clinical_events`, `doctor_patients`, `uploads`, and `label_settings` referencing the old `user_id_hash`.
     3. Checks references in the ML pipeline outputs (e.g. Parquet files) and determines if they need regeneration.

### Summary of Request to Charlie
Please create a **data migration script** to securely transition all legacy users (who were generated under the old email-hash scheme) to purely randomized UUIDv4 identifiers, and ensure all foreign keys in PostgreSQL (and Timescale hypertables) cascade appropriately without downtime.
