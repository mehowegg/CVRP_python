import numpy as np
import random


'''list1 = ['zero', 'one', 'two', 'three', 'four', 'five']
print(list1)

del list1[1]
print(list1)'''

'''numberList = [100, 200, 300, 400]
# Choose elements with different probabilities
sampleNumbers = np.random.choice(numberList, p=[0.10, 0.20, 0.30, 0.40])
print(sampleNumbers)'''


def select_destination(distances, weights, delivered, capacity, demands):
    index, distance = np.nan, np.nan
    for i in range(len(weights)):
        distance = random.choices(distances, weights, k=1)[0]
        print(distance)
        index = distances.index(distance)
        if index not in delivered and demands[index] <= capacity:
            break

    return index, distance


print(select_destination([0,1,2,3,4], [2,3,4,10,3], [2,3], 100, [10,10,10,10,10]))