from collections import Counter
import numpy as np

def mean(xs:list[float]):
    return sum(xs)/len(xs)

def _medianOdd(xs:list[float])->float:
    return sorted(xs)[len(xs)//2]

def _medianEven(xs:list[float])->float:
    sortedList = sorted(xs)
    midpoint = len(sortedList)//2
    return (sortedList[midpoint-1]+sortedList[midpoint])/2 

def median(xs:list[float])->float:
    return _medianEven(xs) if len(xs)%2==0 else _medianOdd(xs)

def mode(xs:list[float])->list[float]:
    countList = Counter(xs)
    max_count = max(countList.values())
    return [xi for xi, count in countList.items() if count==max_count ]
    
def range(xs:list [float])->float:
    return max(xs)- min(xs) 


   
grades = [45,40.3,67.5,90.1,100.0,38,45,84.3,84,84,99,100,67,23]

print(mean(grades))
print(median(grades))
print(mode(grades))
print(range(grades))
print(np.var(grades))
print(np.std(grades))
print(np.median(grades))
