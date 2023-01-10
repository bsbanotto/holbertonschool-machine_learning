#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# your code here
plt.hist(student_grades, facecolor="#348CC4", edgecolor="black",
         bins=range(0, 101, 10), linewidth=0.75)
plt.xlim(0, 100)
plt.xticks(ticks=range(0, 101, 10))
plt.ylim(0, 30)
plt.title("Project A")
plt.ylabel("Number of Students")
plt.xlabel("Grades")
plt.show()
