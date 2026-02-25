import psycopg

DSN = "postgresql://emuser:changeme@127.0.0.1:5432/empredictor"

with psycopg.connect(DSN) as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT conname, pg_get_constraintdef(c.oid) FROM pg_constraint c JOIN pg_class t ON c.conrelid = t.oid WHERE t.relname = 'datapoints'")
        print("Constraints:")
        for row in cur.fetchall(): print(row)
        
        cur.execute("SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'datapoints'")
        print("\nIndexes:")
        for row in cur.fetchall(): print(row)
