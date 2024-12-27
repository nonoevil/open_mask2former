from collections import Counter

a = Counter({"1": 2})
b = Counter({"1": 2})

# 合并
merged = a + b
print(dict(merged))  # 输出: {'1': 2}