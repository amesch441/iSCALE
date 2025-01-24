from PIL import Image
from skimage.transform import resize

from utils import load_pickle
from visual import plot_labels_3d


def make_gif(arrays, filename, duration=100):
    frames = [Image.fromarray(arr) for arr in arrays]
    frame_one = frames[0]
    frame_one.save(
            filename, format='GIF', append_images=frames,
            save_all=True, duration=duration, loop=0)
    print(filename)


def resize_image(img, output_shape):
    img_new = resize(img, output_shape, preserve_range=True)
    img_new = img_new.astype(img.dtype)
    return img_new


def main():

    factor = 50
    duration = 100

    labels = load_pickle('tmp/labels.pickle')
    img = plot_labels_3d(labels)
    img = resize_image(img, (img.shape[0]*factor,) + img.shape[1:])
    make_gif(img[::factor], 'labels-z.gif')
    make_gif(
            img.transpose(1, 2, 0, 3), 'labels-x.gif',
            duration=duration)
    make_gif(
            img.transpose(2, 1, 0, 3), 'labels-y.gif',
            duration=duration)


if __name__ == '__main__':
    main()
