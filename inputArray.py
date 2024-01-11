#OUTPUT
#PS D:\2102040024> C:\Users\Student\AppData\Local\Programs\Python\Python312-32\python.exe .\inputArray.py
#Enter size of the array: 4
#Array
#Enter elements...
#0 : 10
#1 : 20
#2 : 11
#3 : 42
#Entered array:  [10 20 11 42]

import numpy as np

def inputArray(stmt, n):
    print(stmt)
    lst = []
    print("Enter elements...")
    for i in range(n):
        try:
            print(i, ": ", end="")
            ele = int(input())
            lst.append(ele)
        except Exception as e:
            print(e)
    arr = np.array(lst)
    return arr
if __name__ == "__main__":
    print("Entered array: ", inputArray("Array", int(input("Enter size of the array: "))))