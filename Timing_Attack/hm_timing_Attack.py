import sqlite3
import numpy as np
import numpy.core.numeric as NX
import time

# Connect to the database
conn = sqlite3.connect('execution_time.db')
cursor = conn.cursor()

# Create table for the execution time
create_table = "CREATE TABLE IF NOT EXISTS hm_execution_time (mu INTEGER, {})"
table_columns = ', '.join(["hm{} REAL".format(i) for i in range(0, 1000)])
cursor.execute(create_table.format(table_columns))

h=[32,119]
# Loop through the values of mu
for mu in range(0, 1000):
    print("mu = ", mu)
    mu = NX.asanyarray(mu)
    h_value = NX.asarray(np.flip(h))
    iteration = 0
    values = [int(mu)]
    #hm_times = [] #list to store execution times for each hm value
    # Loop through the values of hm
    for hm in range(0, 1000):
        hm = NX.asanyarray(hm)
        hm_start = time.perf_counter() # time start
        h_value = hm * mu
        hm_end = time.perf_counter() # time end
        hm_total_time = hm_end - hm_start
        values.append(hm_total_time)
        if len(values) == 1001:
            cursor.execute("INSERT INTO hm_execution_time VALUES ({})".format(', '.join(['?' for i in range(len(values))])), values)

    # Clear the values for the next iteration
    #print ("value of values before clear = ", values)
    #values = [mu]
    #print ("value of values after clear = ", values)

# Commit the changes and close the connection
conn.commit()
conn.close()