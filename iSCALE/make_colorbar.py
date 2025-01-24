import sys
import matplotlib.pyplot as plt

cmap = sys.argv[1]
vmin = float(sys.argv[2])
vmax = float(sys.argv[3])

fontsize = 80

im = [[vmin, vmax]]
plt.figure(figsize=(2, 10))
plt.imshow(im, cmap=cmap)

plt.gca().set_visible(False)
cax = plt.axes([0, 0, 1, 1])
cbar = plt.colorbar(orientation='vertical', cax=cax)
cbar.ax.tick_params(labelsize=fontsize)
outfile = 'colorbar.png'
plt.savefig(outfile, dpi=300, bbox_inches='tight')
plt.close()
print(outfile)
