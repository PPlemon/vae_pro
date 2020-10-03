import matplotlib.pyplot as plt
num = [14066, 69218, 356150, 38792, 309870, 27864, 652122, 35058, 866738, 57582, 393193, 2, 25702, 8, 261056, 803, 866738, 2360, 241357, 16993391, 11402, 106, 1880173, 150494, 300986, 27772, 57162, 424486, 71246, 309870, 11402, 1987714, 490806, 5301, 90]
# fe = [6, 22, 58, 38, 100, 674, 380, 587, 1374]
activity = ['#', '+', '=', 'l', ']', 'o', '1', 's', ')', 'S', 'N', '8', '/', '7', 'H', 'I', '(', '5', 'n', ' ', 'r', 'P', 'C', '3', '@', '4', '-', '2', 'F', '[', 'B', 'c', 'O', '\\', '6']


fig, ax = plt.subplots(figsize=(20, 10))
# x = range(2, 26, 2)
# ax.plot(x, fe)
# ax.plot(x, co)
# ax.plot(activity, fe, label="fe")
ax.bar(activity, num, width=0.5, label="Number of documents")
# for a, b in zip(activity, num):
#     ax.text(a, b+1, b, ha='center', va='bottom')
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel("char", fontsize=15)
plt.ylabel("numbers", fontsize=15)

ax.legend()
plt.show()
