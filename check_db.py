import psycopg
try:
    conn = psycopg.connect('postgresql://emuser:changeme@localhost:5432/empredictor')
    cur = conn.cursor()
    for table in ['predictions', 'alerts']:
        cur.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table}'")
        cols = cur.fetchall()
        print(f"\nTABLE: {table.upper()}")
        for col in cols:
            print(f"  - {col[0]}: {col[1]}")
    conn.close()
except Exception as e:
    print(f"ERROR: {e}")
