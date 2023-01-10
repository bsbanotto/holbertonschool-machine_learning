#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# your code here
fig = plt.figure(layout="constrained")
spec = fig.add_gridspec(3, 2)


"""Top Left Line Graph"""
ax00 = fig.add_subplot(spec[0, 0])
ax00.plot(y0, color='red')
ax00.set_xlim([0, 10])

"""Top Right Scatter Plot"""
ax01 = fig.add_subplot(spec[0, 1])
ax01.scatter(x1, y1, color='magenta', marker='.')
ax01.set_xlabel('Height (in)', fontsize='x-small')
ax01.set_ylabel('Weight (lbs)', fontsize='x-small')
ax01.set_title("Men's Height vs Weight", fontsize='x-small')

"""Middle Left Semilog Plot"""
ax10 = fig.add_subplot(spec[1, 0])
ax10.semilogy(x2, y2)
ax10.set_xlim(0, 28650)
ax10.set_xlabel("Time (years)", fontsize='x-small')
ax10.set_ylabel("Fraction Remaining", fontsize='x-small')
ax10.set_title("Exponential Decay of C-14", fontsize='x-small')


"""Middle Right Exponential Decay"""
ax11 = fig.add_subplot(spec[1, 1])
ax11.plot(x3, y31, c="red", linestyle="--", label="C-14")
ax11.plot(x3, y32, c="green", label="Ra-226")
ax11.set_xlim(0, 20000)
ax11.set_ylim(0, 1)
ax11.set_title("Exponential Decay of Radioactive Elements", fontsize='x-small')
ax11.set_xlabel("Time (years)", fontsize='x-small')
ax11.set_ylabel("Fraction Remaining", fontsize='x-small')
ax11.legend()

"""Bottom Histogram"""
ax20 = fig.add_subplot(spec[2, :])
ax20.hist(student_grades, facecolor="#348CC4", edgecolor="black",
          bins=range(0, 101, 10), linewidth=0.75)
ax20.set_xlim(0, 100)
ax20.set_xticks(ticks=range(0, 101, 10))
ax20.set_ylim(0, 30)
ax20.set_title("Project A", fontsize='x-small')
ax20.set_ylabel("Number of Students", fontsize='x-small')
ax20.set_xlabel("Grades", fontsize='x-small')

"""Plot Specs"""
fig.suptitle("All in One")
plt.show()
