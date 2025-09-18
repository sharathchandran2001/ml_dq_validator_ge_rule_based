import psycopg2

# === 1. Connection details ===
# Replace with your actual PostgreSQL connection details
conn = psycopg2.connect(
    host="localhost",      # or IP of the server
    port="5432",           # default PostgreSQL port
    database="MA",         # your database name
    user="postgres",       # your PostgreSQL username
    password="admin"  # your PostgreSQL password
)

# === 2. Create a cursor to run queries ===
cur = conn.cursor()

# === 3. Insert data into customer_abc ===
insert_query = """
INSERT INTO public.customer_abc (first_name, last_name, email, dob)
VALUES (%s, %s, %s, %s)
"""
data = ("David", "Miller", "david.miller@example.com", "1992-08-15")
cur.execute(insert_query, data)

# === 4. Copy data from customer_abc into customer_bcd ===
copy_query = """
INSERT INTO public.customer_bcd (first_name, last_name, email, dob)
SELECT first_name, last_name, email, dob
FROM public.customer_abc
WHERE email = %s
"""
cur.execute(copy_query, ("david.miller@example.com",))

# === 5. Commit changes ===
conn.commit()

# === 6. Fetch all data from customer_bcd ===
cur.execute("SELECT * FROM public.customer_bcd")
rows = cur.fetchall()
for row in rows:
    print(row)

# === 7. Close connection ===
cur.close()
conn.close()
