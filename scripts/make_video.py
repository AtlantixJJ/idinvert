import os, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str,
                    help='Path to the inverted data dir.')
parser.add_argument('--type', type=str,
                    help='delta | mix')
args = parser.parse_args()


DIR = args.dir
FPS = 24

if args.type == 'delta':
  for i in range(16):
    # combined video
    basecmd = f"ffmpeg -i {DIR}/enc_{i}.mp4 -i {DIR}/ori_{i}.mp4 -i {DIR}/rem_{i}.mp4 -filter_complex \"[0:v:0][1:v:0][2:v:0]hstack=inputs=3\" -b:v 16000k -y {DIR}/delta_final_{i}.mp4"
    os.system(basecmd) 
elif args.type == 'mix':
  for i in range(16):
    videos = [f"{DIR}/mix_{i}_mix{l}.mp4" for l in [1, 2, 3, 4, 6, 8, 10, 12]]
    videos = " -i  ".join(videos)

    # combined video
    basecmd = f"ffmpeg -i {videos} -filter_complex \"[0:v][1:v][2:v][3:v]hstack=inputs=4[r1];[4:v][5:v][6:v][7:v]hstack=inputs=4[r2];[r1][r2]vstack=inputs=2\" -b:v 16000k -y {DIR}/mix_final_{i}.mp4"
    os.system(basecmd) 
elif args.type == 'mixafter':
  for i in range(16):
    videos = [f"{DIR}/mixafter_{i}_mix{l}.mp4" for l in [1, 2, 3, 4, 6, 8, 10, 12]]
    videos = " -i  ".join(videos)

    # combined video
    basecmd = f"ffmpeg -i {videos} -filter_complex \"[0:v][1:v][2:v][3:v]hstack=inputs=4[r1];[4:v][5:v][6:v][7:v]hstack=inputs=4[r2];[r1][r2]vstack=inputs=2\" -b:v 16000k -y {DIR}/mixafter_final_{i}.mp4"
    os.system(basecmd) 
elif args.type == 'at':
  for i in range(16):
    videos = [f"{DIR}/at{n}_{i}.mp4" for n in [4, 5]]
    videos = " -i  ".join(videos)

    # combined video
    basecmd = f"ffmpeg -i {videos} -filter_complex \"[0:v][1:v]hstack\" -b:v 16000k -y {DIR}/at45_final_{i}.mp4"
    os.system(basecmd) 

elif args.type == 'MPI':
  indir = args.dir
  for i in range(1):
    basecmd = f"ffmpeg -i {indir}/{i:06d}_transform%03d_inv.png -b:v 16000k -y {indir}/{i:06d}_inv.mp4"
    os.system(basecmd)  

elif args.type == 'MPI_noenc':
  indir = args.dir
  for i in range(1):
    basecmd = f"ffmpeg -i {indir}/{i:06d}_transform%03d_inv_noenc.png -b:v 16000k -y {indir}/{i:06d}_inv_noenc.mp4"
    os.system(basecmd)  
   
# ffmpeg -i enc.mp4 -i rem.mp4 -filter_complex "[0:v:0][1:v:0]hstack=inputs=2" -b:v 16000k -y final.mp4
