import hashlib
h = hashlib.md5("123".encode('utf-8')).hexdigest()
print(h)
