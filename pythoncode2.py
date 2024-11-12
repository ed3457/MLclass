my_list= [1 , 2]
my_tuple = (1 , 2,3,4)
other_tuple = 3 , 4
my_list [1] = 3 # my_list is now [1, 3]
try:
    my_tuple [1] = 3
except TypeError:
    print("cannot modify a tuple")

#sum=0
#for i in my_tuple:
#    sum+=i
#print (sum)

# write code to reverse a tuple 

new_tuple = my_tuple[::-1]
print(new_tuple)


grades= {"Joel": 80 , "Tim": 95, "James": 100 } # dictionary
if "Joel" in grades:
    print(grades["Joel"])
else:
    print("Key does not exist!")

# write code to find the average grade for this class
all_values =grades.values ()

grade_sum = 0
for v in all_values:
    grade_sum+=v
print(grade_sum/len(all_values))

avg_grade = sum(all_values)/len(grades)
print(avg_grade)


user_input =input("Please provide a name:")

while True:
    if user_input =="":
        print("Value is empty! Please provide a name:")
        user_input =input("Please provide a name:")
    else:

        print("Input is good!")
        break

square_dict={ x : x * x for x in range (5)} 
print(square_dict)

print(5/3)
print(5//3)