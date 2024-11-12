import re
regex = r'\d{3}-\d{3}-\d{4}'

with open('sampletext.txt', 'r') as file:
  
    for row in file:
        phone_numbers = re.findall(regex, row)
        for number in phone_numbers:
            print(number)



