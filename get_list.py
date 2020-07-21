import sys
fpath = sys.argv[1]
name = fpath[fpath.rfind("/")+1:]
f = open(f"{name}.list", "w")

for i in range(16):
  for j in range(8000):
    op = 0
    if j % 800 == 0:
      op = 200
    f.write(f"{fpath}/{i:06d}_transform{j:03d}.png {op}\n")
f.close()

