from sklearn.decomposition import PCA
from tqdm import tqdm
from PIL import Image
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib import cm
cmap = cm.get_cmap("plasma")
import matplotlib.style as style
from mpl_toolkits.mplot3d import Axes3D
style.use('seaborn-poster') #sets the size of the charts
style.use('ggplot')
colors = list(matplotlib.colors.cnames.keys())

from moviepy.editor import VideoClip
import numpy as np
import glob


def fig2data(fig):
    """
    fig = plt.figure()
    image = fig2data(fig)
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()
 
    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    image = np.asarray(image.convert("RGB"))
    return image


FPATH = "results/inversion/stylegan_bedroom_transform/encoded_codes.npy"
data = np.load(FPATH)#np.random.randn(8000, 14, 512) #np.load(FPATH)
DSIZE = 800
NSIZE = DSIZE * 2 * 5
N = data.shape[0] // NSIZE

print(f"=> Full data shape: {data.shape}")
print(f"=> Number of image: {N}")

def pca(feats):
  model = PCA().fit(feats - feats.mean(0, keepdims=True))
  return model


def plot(fname, cords, N=-1):
    c = cmap(np.linspace(0, 1, cords.shape[0]))
    if N > 0:
      G = cords.shape[0] // N
      c = cmap(np.array([[i] * G for i in range(N)]) / float(N))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f'3D {exp_ratio[2]:.3f}')
    ax.scatter(cords[:, 0], cords[:, 1], cords[:, 2], s=1, c=c)
    azim, elev = ax.azim, ax.elev
    FPS = 16
    number = 120
    def make_frame(t):
        proc = t * FPS / number
        ax.view_init(azim + proc * 30, elev + proc * 360)
        return fig2data(fig)

    animation = VideoClip(make_frame, duration=number / FPS)
    animation.write_videofile(f"{fname}_3droll.mp4", fps=FPS)
    plt.close()


dic_i = {i:{} for i in range(N)}
dic_d = {i:{} for i in range(10)}
# for each image
for i in range(N):
  imid = NSIZE * i
  # for each changin direction
  for j in range(10):
    dic_i[i][j] = {}
    segid = DSIZE * j
    x = data[imid + segid : imid + segid + DSIZE].reshape(-1, 14 * 512)
    model = pca(x)
    exp_ratio = np.cumsum(model.explained_variance_ratio_[:3])
    components = model.components_[:, :3]
    dic_i[i][j]["exp_ratio"] = exp_ratio
    dic_i[i][j]["components"] = components
    if i <= 2:
      cord = model.transform(x)
      plot(f"image{i}_direction{j}", cord)

# for each changing direction
for j in range(10):
  segid = DSIZE * j
  xs = []
  for i in range(N):
    imid = NSIZE * i
    xs.append(data[imid + segid : imid + segid + DSIZE])
  xs = np.concatenate(xs, 0).reshape(-1, 14 * 512)
  model = pca(xs)
  exp_ratio = np.cumsum(model.explained_variance_ratio_[:3])
  components = model.components_[:, :3]
  dic_d[j]["exp_ratio"] = exp_ratio
  dic_d[j]["components"] = components
  plot(f"direction{j}", model.transform(xs))

np.save("stylegan_bedroom_image_dic.npy", dic_i)
np.save("stylegan_bedroom_direction_dic.npy", dic_d)
