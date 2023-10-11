import psycopg2


conn = psycopg2.connect(
 host="socs-db.dcs.gla.ac.uk",
 user="lev3_20_anagnosc",
 password="anagnosc",
 dbname="postgres"
)
print(conn)
conn.close()
