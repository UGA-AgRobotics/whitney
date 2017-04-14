import matplotlib.pyplot as plt
from skimage import io, filters, morphology, exposure, color, feature, segmentation, graph, draw
from skimage.feature import blob_doh
from scipy import ndimage as ndi
import numpy as np


def test_lines(img):
    gauss = filters.gaussian(img, sigma=2)
    equalize = exposure.equalize_adapthist(gauss)
    # otsu = filters.threshold_otsu(equalize)
    # thresh = gauss <= otsu
    # close = morphology.binary_closing(thresh)
    # opening = morphology.binary_opening(close)
    edges = feature.canny(equalize, sigma=4)
    return edges


def test_thresh(img):
    gauss = filters.gaussian(img, sigma=2)
    equalize = exposure.equalize_adapthist(gauss)
    fig, ax = filters.try_all_threshold(equalize)


if __name__ == '__main__':

    image = io.imread("../img/cotton1.png")

    image_gray = color.rgb2gray(image)
    image_gray = exposure.equalize_adapthist(image_gray)
    out = filters.rank.tophat(image_gray, morphology.disk(10))
    close = morphology.closing(out)
    opening = morphology.opening(close)
    bilat_img = filters.rank.mean_bilateral(opening, morphology.disk(30), s0=10, s1=10)

    blobs_doh = blob_doh(bilat_img, min_sigma=20, max_sigma=50, num_sigma=20, threshold=.006)

    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True,
                           subplot_kw={'adjustable': 'box-forced'})

    ax.imshow(image, interpolation='nearest', cmap='gray')
    for blob in blobs_doh:
        y, x, r = blob
        c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
        rr, cc = draw.circle(y, x, r, shape=image.shape)

        ax.add_patch(c)

    ax.text(1, 1, "Bolls: "+str(len(blobs_doh)), color='white',
            bbox={'facecolor': 'black', 'alpha': 1, 'pad': 2})
    ax.set_axis_off()

    plt.tight_layout()
    plt.show()

    # r = image[..., 0]
    # g = image[..., 1]
    # b = image[..., 2]
    # gauss = filters.gaussian(image_gray, sigma=0.5)
    # equalize = exposure.equalize_adapthist(gauss)
    # otsu = filters.threshold_otsu(equalize)
    # thresh = gauss <= otsu
    # thresh = 1 - thresh
    # close = morphology.binary_closing(thresh)
    # opening = morphology.binary_opening(close)
    # plt.imshow(opening, cmap='gray')

    # plt.figure(1)
    # plt.imshow(test_process(r))
    # plt.figure(2)
    # plt.imshow(test_process(g))
    # plt.figure(3)
    # plt.imshow(test_process(b))

    # plt.show()
