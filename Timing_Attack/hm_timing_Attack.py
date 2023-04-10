import sqlite3
import numpy as np
import numpy.core.numeric as NX
import time

# Connect to the database
conn = sqlite3.connect('execution_time.db')
cursor = conn.cursor()

# Create table for the execution time
create_table = "CREATE TABLE IF NOT EXISTS hm_32template (mu INTEGER, {} )"
table_columns = ', '.join(["hm{}_{} REAL".format(h0, h1) for h0 in range (32, 33) for h1 in range(0, 200)])
cursor.execute(create_table.format(table_columns))

#h=[32,119]

for mu in range (0,1000):
    values = [int(mu)]
    for h0 in range (32,33):
        for h1 in range (0,200):
            h_value = NX.asarray(np.flip([h0,h1]))
            if isinstance(mu, np.poly1d):
                hm = 0
            else:
                mu = NX.asanyarray(mu)
                hm = NX.zeros_like(mu)
            hm_start = time.perf_counter() # time start
            for pv in h_value:
                hm = hm * mu + pv
            hm_end = time.perf_counter() # time end
            hm_total_time = hm_end - hm_start
            values.append(hm_total_time)
            print(f'for mu{mu} and key {h0}-{h1} time = {hm_total_time}')
            if len(values) == 201:
                cursor.execute("INSERT INTO hm_32template VALUES ({})".format(', '.join(['?' for i in range(len(values))])), values)
            hm_total_time = 0

# Commit the changes and close the connection
conn.commit()
conn.close()


