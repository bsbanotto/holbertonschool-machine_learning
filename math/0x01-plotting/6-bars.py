#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

# your code here
apples = fruit[0]
bananas = fruit[1]
oranges = fruit[2]
peaches = fruit[3]
print("apples = ", apples)
print("bananas = ", bananas)
print("oranges = ", oranges)
print("peaches = ", peaches)
labels = ['Farrah', 'Fred', 'Felicia']
plt.ylim(0, 80)
plt.title("Number of Fruit per Person")
plt.ylabel("Quantity of Fruit")
plt.bar(labels, fruit[0][:], color='red', label='apples')
plt.bar(labels, fruit[1][:], color='yellow', bottom=apples, label='bananas')
plt.bar(labels, fruit[2][:], color='#FF8000', bottom=apples + bananas,
        label='oranges')
plt.bar(labels, fruit[3][:], color='#FFE5B4',
        bottom=apples + bananas + oranges, label='peaches')
plt.legend()
plt.show()
