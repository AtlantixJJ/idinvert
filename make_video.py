import os, sys
DIR = sys.argv[1]
UDIR = DIR.replace("results/inversion", "../synsin/mydata")
FPS = 24

for i in range(2):
  # original video
  basecmd = f"ffmpeg -r {FPS} -i {UDIR}/{i:06d}_transform%02d0.png -b:v 16000k -y {i:06d}_ori.mp4"
  os.system(basecmd)

  # edited video
  for t in ["enc", "rem"]:
    basecmd = f"ffmpeg -r {FPS} -i {DIR}/{i:06d}_transform%02d0_{t}.png -b:v 16000k -y {i:06d}_{t}.mp4"
    os.system(basecmd)
  
  # combined video
  basecmd = f"ffmpeg -i {i:06d}_ori.mp4 -i {i:06d}_enc.mp4 -i {i:06d}_rem.mp4 -filter_complex \"[0:v:0][1:v:0][2:v:0]hstack=inputs=3\" -b:v 16000k -y {i:06d}_final.mp4"
  os.system(basecmd) 
  
  
