import numpy as np
import numpy.core.numeric as NX
import sqlite3
import time

# Connect to the database
conn = sqlite3.connect('data_time.db')
cursor = conn.cursor()

# Create tables to store the data
# cursor.execute("""CREATE TABLE IF NOT EXISTS fm_message_table (mu INTEGER, execution_time REAL)""")
cursor.execute("""CREATE TABLE IF NOT EXISTS hm_message_table (mu INTEGER, execution_time REAL)""")

mu = 251
f=[11,207]
h=[32,119]


h_value = NX.asarray(np.flip(h))
for mu in range(0, 1000):
     # mu = NX.asanyarray(mu)
     hm = NX.zeros_like(mu)
     hm_start = time.perf_counter() # time start
     for pv in h_value:
          hm = hm * mu + pv
     hm_end = time.perf_counter() # time end
     hm_total_time = hm_end - hm_start
     cursor.execute("""INSERT INTO hm_message_table (mu, execution_time) VALUES (?, ?)""", (mu, hm_total_time))
     conn.commit() # save changes to the database
     hm_total_time = 0

conn.commit()
conn.close()