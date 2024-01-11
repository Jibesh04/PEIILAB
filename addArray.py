#OUTPUT
#PS D:\2102040024> C:\Users\Student\AppData\Local\Programs\Python\Python312-32\python.exe .\addArray.py
#Enter array size: 5
#Array 1
#Enter elements...
#0 : 15
#1 : -5
#2 : 22
#3 : 9
#4 : 18
#Array 2
#Enter elements...
#0 : 7
#1 : 14
#2 : -15
#3 : 9
#4 : -18
#Adding [15 -5 22  9 18] with [  7  14 -15   9 -18] ...
#Resulting array: [22  9  7 18  0]

from inputArray import inputArray
import numpy as np

def addArray(a1, a2):
    print("Adding", a1, "with", a2, "...")
    return np.add(a1, a2)

if __name__ == "__main__":
    n = int(input("Enter array size: "))
    a1 = inputArray("Array 1", n)
    a2 = inputArray("Array 2", n)
    print("Resulting array:", addArray(a1, a2))
