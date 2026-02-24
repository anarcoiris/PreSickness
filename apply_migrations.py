"""
Apply SQL migrations to the database.
Usage: python apply_migrations.py
"""
import asyncio
import sys
import os
from pathlib import Path

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "empredictor")
DB_USER = os.getenv("DB_USER", "emuser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "changeme")
DSN = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


async def apply_migration(migration_file: Path):
    """Apply a single migration file."""
    import psycopg
    from psycopg.rows import dict_row
    
    print(f"Applying migration: {migration_file.name}")
    
    sql_content = migration_file.read_text(encoding='utf-8')
    
    async with await psycopg.AsyncConnection.connect(DSN, row_factory=dict_row) as conn:
        try:
            await conn.execute(sql_content)
            await conn.commit()
            print(f"✓ Migration {migration_file.name} applied successfully")
            return True
        except Exception as e:
            print(f"✗ Migration {migration_file.name} failed: {e}")
            return False


async def main():
    migrations_dir = Path(__file__).parent / "migrations"
    if not migrations_dir.exists():
        print(f"Migrations directory not found: {migrations_dir}")
        return
    
    # Get all .sql files sorted by name
    migration_files = sorted(migrations_dir.glob("*.sql"))
    
    if not migration_files:
        print("No migration files found")
        return
    
    print(f"Found {len(migration_files)} migration(s)")
    print(f"Database: {DB_HOST}:{DB_PORT}/{DB_NAME}")
    print("-" * 50)
    
    success_count = 0
    for mf in migration_files:
        if await apply_migration(mf):
            success_count += 1
    
    print("-" * 50)
    print(f"Applied {success_count}/{len(migration_files)} migrations")


if __name__ == "__main__":
    asyncio.run(main())
