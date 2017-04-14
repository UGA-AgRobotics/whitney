import matplotlib.pyplot as plt
from skimage import io, filters, morphology, exposure, color, feature, segmentation, graph, draw, measure
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

    image = io.imread("../img/cotton1.jpg")
    image_gray = color.rgb2gray(image)
    bilat_img = filters.rank.mean_bilateral(image_gray, morphology.disk(30), s0=10, s1=10)
    eq = exposure.equalize_adapthist(bilat_img)
    out = filters.rank.tophat(eq, morphology.disk(10))
    close = morphology.closing(out)
    opening = morphology.opening(close)
    opening = 255 - opening

    blobs_doh = blob_doh(opening, min_sigma=10, max_sigma=100, num_sigma=10, threshold=.006)

    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True,
                           subplot_kw={'adjustable': 'box-forced'})

    ax.imshow(image, interpolation='nearest', cmap='gray')
    labels = np.zeros(image_gray.shape, dtype=np.int)
    thresholds = []
    for index, blob in enumerate(blobs_doh, start=1):
        y, x, r = blob
        rr, cc = draw.circle(y, x, r, shape=image.shape)
        labels[rr, cc] = index
        thresholds.append(filters.threshold_otsu(bilat_img[rr, cc]))

    threshold = np.percentile(thresholds, 65)

    count = 0
    for index, blob in enumerate(blobs_doh, start=1):
        y, x, r = blob
        if thresholds[index-1] <= threshold:
            continue
        count = count + 1
        c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
        ax.text(x, y, index, color='white',
                bbox={'facecolor': 'black', 'alpha': 0.5, 'pad': 1})
        ax.add_patch(c)

    ax.text(1, 1, "Bolls: "+str(count), color='white',
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
