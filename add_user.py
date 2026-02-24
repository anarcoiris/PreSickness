import psycopg
conn = psycopg.connect('postgresql://emuser:changeme@localhost:5432/empredictor')
cur = conn.cursor()
cur.execute("INSERT INTO users (user_id_hash, email, name, status) VALUES ('paciente1', 'paciente1@example.com', 'Paciente 1', 'active') ON CONFLICT DO NOTHING")
conn.commit()
conn.close()
print("User 'paciente1' ensured in DB.")
