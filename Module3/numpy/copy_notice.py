# 当需要复制np array的时候的注意事项：
# b 和 a 指向同一个数组，当其中一个受到了更改，另一个的值也受到了更改
import numpy as np
a = np.array([1,2,3])
b = a
b[0] = 100

print(f'a array: {a}')
print(f'b array: {b}')


# 所以在复制数组的时候（如果不想更改一个数组的同时更改另一个数组），
# 需要用到以下方法 `.copy()`

import numpy as np
a = np.array([1,2,3])
b = a.copy()
b[0] = 100

print("\nUse .copy()")
print(f'a array: {a}')
print(f'b array: {b}')
