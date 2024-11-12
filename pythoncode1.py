def avg(x,y):
    sum=0
    for i in range(x,y+1):
        sum+=i
    print (sum)
    avg=sum/(y-x+1)
    print(avg)

def oddOrEven(name):
    length=len(name)
    if (length % 2==0):
        print( f"{name} is Even")
    else:
         print( f"{name} is Odd")
#avg(1,100)
#avg(100,1000)

name = "Joell"
# find out if the string has odd or even number of chars
#oddOrEven(name)

list1=[1,2,3,4]
#print(sum(list1)/len(list1))

#for i in list1:
#    print(i)

print(list1[:-2])







