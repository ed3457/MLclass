import pyodbc
import pandas as pd

# Database connection details
server = r'.\sqlexpress' 
database = 'northwind' 

# Connection string
conn_str = f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};Trusted_Connection=yes;'

# Connect to the database
conn = pyodbc.connect(conn_str)

# Create a cursor object
cursor = conn.cursor()

# Execute a query
query = "SELECT * FROM employees"
cursor.execute(query)

# Fetch the results
rows = cursor.fetchall()

print(rows)
# Convert the results to a pandas DataFrame (optional)

# Close the connection
conn.close()

