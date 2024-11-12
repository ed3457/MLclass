# Open the file in read mode
with open('fruit_data_with_colors.txt', 'r') as file:
    # Read the contents of the file
    content = file.readline()

# Print the content
print(content)
import pandas as pd
import csv
data = []
with open('fruit_data_with_colors.txt') as f:
    tab_reader = csv.reader(f, delimiter='\t')
    for row in tab_reader:
        label = row[0]
        symbol = row[1]
        color_score = row[2]
        #print(row)
        data.append([label, symbol, color_score])
df = pd.DataFrame(data, columns=['Label', 'Symbol', 'color score'])

# Display the DataFrame
print(df.describe())