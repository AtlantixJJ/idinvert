import sys
fpath = sys.argv[1]
t = sys.argv[2]
name = fpath[fpath.rfind("/")+1:]


if t == '1':
  f = open(f"{name}.list", "w")
  for i in range(16):
    for j in range(800):
      op = 0
      if j == 0:
        op = 200
      f.write(f"{fpath}/{i:06d}_transform{j:03d}.png {op}\n")
  f.close()
if t == '2':
  f = open(f"{name}_single.list", "w")
  for j in range(800):
    op = 1000
    f.write(f"{fpath}/000000_transform{j:03d}.png {op}\n")
  f.close()

