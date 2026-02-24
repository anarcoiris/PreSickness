-- Add user_oauth table for storing OAuth tokens
CREATE TABLE IF NOT EXISTS user_oauth (
    id SERIAL PRIMARY KEY,
    user_id_hash VARCHAR(64) NOT NULL REFERENCES users(user_id_hash) ON DELETE CASCADE,
    provider VARCHAR(50) NOT NULL, -- e.g. 'google'
    access_token TEXT,
    refresh_token TEXT,
    token_expiry TIMESTAMPTZ,
    scope TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(user_id_hash, provider)
);

CREATE INDEX idx_user_oauth_user ON user_oauth(user_id_hash);
