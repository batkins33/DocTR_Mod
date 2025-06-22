import re

s = "7:55/7 627843"
m = re.search(r"\d{6}", s)
print(m.group(0))  # Should print 627843
