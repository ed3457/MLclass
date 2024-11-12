import requests
import pandas as pd

# URL of the public API that provides user data
api_url = 'https://jsonplaceholder.typicode.com/users'

# Send a GET request to the API
response = requests.get(api_url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    data = response.json()

    # Convert the JSON data into a Pandas DataFrame
    df = pd.json_normalize(data)
    
    # Print the DataFrame
    print(df.head())  # Display the first few rows
    print(df.describe())
else:
    print(f"Failed to retrieve data. Status code: {response.status_code}")




    
