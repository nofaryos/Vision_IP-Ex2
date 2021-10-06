# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    print_hi('PyCharm')
    a = np.array([1,2,3])
    b = np.array([[5,5], [5,5]])
    print((b*a[0:2]).sum())
    a = np.array([[1,2,3],[1,2,3],[1,2,3]])
    a[0:2, 0:2] = b
    print("a", a)
    b = np.array([[1],[2],[3]])
    print(a*b)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
