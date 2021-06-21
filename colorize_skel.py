# CS194-26 (CS294-26): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import skimage as sk
import skimage.io as skio
import skimage.transform
import skimage.filters
from skimage import feature

# name of the input file
imname = 'data/emir.tif'

# read in the image
im = skio.imread(imname)

print("dimension: ", im.shape)
# convert to double (might want to do this later on to save memory)    
im = sk.img_as_float(im)

# skio.imshow(im)
# skio.show()

# compute the height of each part (just 1/3 of total)
height = im.shape[0] // 3
width = im.shape[1]
print("height: ", height)
# separate color channels
b = im[:height]
g = im[height: 2 * height]
r = im[2 * height: 3 * height]

canny = feature.canny(g) * 1
diff = np.linalg.norm(canny - np.roll(canny, 1, axis=0), axis=0) / height
print(canny[:10, :10])
print(diff[450:460])
print(np.mean(canny, axis=0)[450:460])


# skio.imshow_collection([b, feature.canny(b)])
# skio.show()
def SSD(im1, im2):
    try:
        return -np.linalg.norm(np.abs(np.gradient(im1)[0]) - np.abs(np.gradient(im2)[0])) - np.linalg.norm(
            np.abs(np.gradient(im1)[1]) - np.abs(np.gradient(im2)[1]))
    except ValueError:
        return -np.linalg.norm(im1 - im2)


def NCC(im1, im2, eps=1e-7):
    try:
        grad1, grad2 = np.gradient(im1), np.gradient(im2)
        l = grad1[0].size
        grad1x, grad2x = grad1[0].reshape((l,)), grad2[0].reshape((l,))
        grad1y, grad2y = grad1[1].reshape((l,)), grad2[1].reshape((l,))

        grad1x, grad2x = grad1x / (np.linalg.norm(grad1x) + eps), grad2x / (np.linalg.norm(grad2x) + eps)
        grad1y, grad2y = grad1y / (np.linalg.norm(grad1y) + eps), grad2y / (np.linalg.norm(grad2y) + eps)
        return np.abs(grad1x).dot(np.abs(grad2x)) + np.abs(grad1y).dot(np.abs(grad2y))
    except ValueError:
        im1, im2 = im1.reshape((im1.size,)), im2.reshape((im2.size,))
        im1_normalized, im2_normalized = im1 / np.linalg.norm(im1), im2 / np.linalg.norm(im2)
        return im1_normalized.dot(im2_normalized)


def NCC2(arr1, arr2):
    arr1 = arr1.reshape((arr1.size,))
    arr2 = arr2.reshape((arr2.size,))
    return arr1.dot(arr2)


def pyramid_find_displacement(im, fixed, n, e, matching_metric=NCC):
    if n == 0:
        return find_displacement(im, fixed)
    im_resized = sk.transform.rescale(sk.filters.gaussian(im), .5)
    fixed_resized = sk.transform.rescale(sk.filters.gaussian(fixed), .5)
    displacement_vector = pyramid_find_displacement(im_resized, fixed_resized, n - 1, e * 2) * 2
    print("displacement vector: ", displacement_vector)
    print("shape: ", im.shape)
    return find_neighboring_displacement(im, fixed, displacement_vector, e, matching_metric)


def find_neighboring_displacement(im, fixed, displacement_vector, e, matching_metric=NCC):
    copy = np.copy(im)
    best_displacement_x = displacement_vector[0]
    best_displacement_y = displacement_vector[1]
    best_matching_score = float("-inf")
    h, w = im.shape
    x_lower_bound = max(best_displacement_x - e, -h // 10 + 1)
    x_upper_bound = min(best_displacement_x + e, h // 10 - 1) + 1
    y_lower_bound = max(best_displacement_y - e, -w // 10 + 1)
    y_upper_bound = min(best_displacement_y + e, w // 10 - 1) + 1
    for i in range(x_lower_bound, x_upper_bound):
        for j in range(y_lower_bound, y_upper_bound):
            im = np.roll(np.roll(copy, j, axis=1), i, axis=0)
            score = matching_metric(fixed, im)
            if score > best_matching_score:
                best_displacement_x, best_displacement_y, best_matching_score = i, j, score
    # print("\ti:{}, j, {}, x: [{}, {}], {}; y: [{},  {}], {}, score: {}, best score: {}".format(i, j,
    # x_lower_bound, x_upper_bound, best_displacement_x, y_lower_bound, y_upper_bound, best_displacement_y, score,
    # best_matching_score))

    return np.array([best_displacement_x, best_displacement_y])


def find_displacement(im, fixed, matching_metric=NCC):
    copy = np.copy(im)
    best_displacement_x = 0
    best_displacement_y = 0
    best_matching_score = float("-inf")
    h, w = im.shape
    for i in range(-h + 1, h):
        for j in range(-w + 1, w):
            im = np.roll(np.roll(copy, j, axis=1), i, axis=0)
            score = matching_metric(fixed, im)
            if score > best_matching_score:
                # print("\t{}, {}, score: {}".format(i, j, score))
                best_displacement_x, best_displacement_y, best_matching_score = i, j, score
    return np.array([best_displacement_x, best_displacement_y])


def pyramid_align(im, fixed, matching_metric=NCC):
    # skio.imshow_collection(np.gradient(im))
    # print(np.mean(np.gradient(im)[0]))
    # skio.show()
    n = int(np.floor(np.log2(np.min(im.shape))))
    displacement_x, displacement_y = pyramid_find_displacement(im, fixed, n, 1, matching_metric)
    print(displacement_x, displacement_y)
    return np.roll(np.roll(im, displacement_y, axis=1), displacement_x, axis=0)


def crop(r, g, b):
    for i in range(4):
        r, g, b = crop_one_side(r, g, b)
        r, g, b = np.rot90(r), np.rot90(g), np.rot90(b)
    return r, g, b


# def crop_one_side(r, g, b):
# 	# r_gradient, g_gradient, b_gradient = get_blockwise_gradient(r), get_blockwise_gradient(g), get_blockwise_gradient(b)
# 	# r_gradient, g_gradient, b_gradient = np.gradient(r)[0], np.gradient(g)[0], np.gradient(b)[0]
# 	# r_gradient, g_gradient, b_gradient = difference_of_gaussians(r), difference_of_gaussians(g), difference_of_gaussians(b)
# 	r_gradient, g_gradient, b_gradient = r, g, b
# 	# print(r_gradient[:50,50])
# 	# print(np.mean(r_gradient[:,50]))
# 	# print(np.max(np.abs(b_gradient[:,5])))
# 	row_to_crop = max(find_row_to_crop(r_gradient), find_row_to_crop(g_gradient), find_row_to_crop(b_gradient))
# 	print("row to crop: ", row_to_crop)
# 	r_cropped, g_cropped, b_cropped = r[:,row_to_crop:], g[:,row_to_crop:], b[:,row_to_crop:]

# 	row_to_crop = max(find_row_to_crop_black(r_cropped), find_row_to_crop_black(g_cropped), find_row_to_crop_black(b_cropped))
# 	print("row to crop: ", row_to_crop)
# 	r_cropped, g_cropped, b_cropped = r_cropped[:,row_to_crop:], g_cropped[:,row_to_crop:], b_cropped[:,row_to_crop:]

# 	return r_cropped, g_cropped, b_cropped

# def find_row_to_crop(im, thresh=.90):
# 	start = 0
# 	h, w = im.shape
# 	end = w
# 	m = 8 #(start + end) // 2
# 	blocksize = 8
# 	# while (m < w and np.mean(np.abs(im[:,m])) < thresh):
# 	while (m < w // 10):
# 		if np.mean(np.abs(im[:,m:m+blocksize])) < thresh:
# 			if blocksize == 1:
# 				break
# 			else:
# 				blocksize -= 1
# 		else:
# 			m += blocksize
# 		# print(np.mean(np.abs(im[:,m:m+blocksize])))
# 	return m

def crop_one_side(r, g, b):
    r_gradient, g_gradient, b_gradient = r, g, b

    row_to_crop = max(find_row_to_crop(r_gradient), find_row_to_crop(g_gradient), find_row_to_crop(b_gradient))
    print("row to crop: ", row_to_crop)
    r_cropped, g_cropped, b_cropped = r[:, row_to_crop:], g[:, row_to_crop:], b[:, row_to_crop:]

    return r_cropped, g_cropped, b_cropped


def find_row_to_crop(im, thresh=.90):
    m = 0
    _, w = im.shape
    end = w // 10
    canny = feature.canny(im) * 2 - 1
    h, _ = canny.shape
    left_width = 4
    line_width = 2
    right_width = 1
    mask_width = left_width + line_width + right_width
    mean = np.mean(canny, axis=1)
    mask = np.concatenate((-1 * np.ones((h, left_width)), np.ones((h, line_width)), -1 * np.ones((h, right_width))),
                          axis=1)
    # mask = np.concatenate((-1 * np.ones((h, left_width)), np.ones((h, line_width)), -1 * np.ones((h, right_width))), axis=1)
    border = 0
    best_score = float("-inf")
    print(np.sum(canny[:, line_width + right_width]))
    print(canny.shape, im.shape)

    # print(canny[:20,:20])
    print(best_score)
    scores = []
    while (m < end):
        score = NCC2(mask, canny[:, m: m + mask_width])
        scores.append(score)
        if (score > best_score):
            best_score = score
            border = m + left_width
        print("best score: ", best_score, border, "score: ", score, m)
        m += 1
    scores = np.asarray(scores)
    std = np.std(scores)
    mean = np.mean(scores)
    print("sd: ", std, "mean: ", mean)
    local_maxima_indices = np.where(scores >= mean + std)[0]
    print("local maxima:", local_maxima_indices)
    if len(local_maxima_indices) > 0 and np.max(local_maxima_indices) + left_width > border:
        border = np.max(local_maxima_indices) + left_width
    # skio.imshow_collection([(canny + 1) / 2, (canny[:,border:] + 1) / 2])
    # skio.show()
    return border


# def find_row_to_crop(im, thresh=.90):
# 	m = 0
# 	h, w = im.shape
# 	end = w // 5
# 	canny = feature.canny(b) * 1
# 	diff = np.linalg.norm(canny - np.roll(canny, 1, axis=0), axis=0) / h
# 	mean = np.mean(canny, axis=0)
# 	border_found = False
# 	border = m
# 	while m < end:
# 		while m < end:
# 			if mean[m] > .06 and diff[m] < .002:
# 				border_found = True
# 				border = m
# 				break
# 			else:
# 				m += 1
# 		if border_found:
# 			border_found = False
# 		else:
# 			break
# 		m += 1
# 	return border

def find_row_to_crop_black(im, thresh=.2):
    start = 0
    h, w = im.shape
    end = w
    m = 0  # (start + end) // 2
    blocksize = 8
    while (m < w // 10):
        if np.mean(np.abs(im[:, m:m + 8])) > thresh:
            if blocksize == 1:
                break
            else:
                blocksize -= 1
        else:
            m += blocksize
        print(m, np.mean(np.abs(im[:, m:m + blocksize])))
        m += blocksize
    return m


def difference_of_gaussians(im, num_differences=5):
    sigmas = np.linspace(0, 2, 2 * num_differences)
    difference = np.zeros(im.shape)
    for i in range(num_differences):
        sigma1 = sigmas[2 * i]
        sigma2 = sigmas[2 * i + 1]
        difference += sk.filters.gaussian(im, sigma=sigma1) - sk.filters.gaussian(im, sigma=sigma2)
    return difference / num_differences * 100


def get_luminance(r, g, b):
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def get_brightest_pixel(channel):
    brightest_index = np.argmax(channel)
    h, w = channel.shape
    x, y = brightest_index // h, brightest_index % w
    return x, y


def get_darkest_pixel(channel):
    darkest_index = np.argmin(channel)
    h, w = channel.shape
    x, y = darkest_index // h, darkest_index % w
    return x, y


def auto_contrast(r, g, b):
    luminance = get_luminance(r, g, b)
    brightest_x, brightest_y = get_brightest_pixel(luminance)
    darkest_x, darkest_y = get_darkest_pixel(luminance)
    brightest_channel_index = np.argmax([r[brightest_x, brightest_y],
                                         g[brightest_x, brightest_y],
                                         b[brightest_x, brightest_y]])
    print(brightest_channel_index)
    brightest_channel = [r, g, b][brightest_channel_index]
    darkest_channel_index = np.argmin([r[darkest_x, darkest_y],
                                       g[darkest_x, darkest_y],
                                       b[darkest_x, darkest_y]])
    print(darkest_channel_index)
    darkest_channel = [r, g, b][darkest_channel_index]
    darkest_channel = normalize_brightness(darkest_channel)
    result = [r, g, b]
    result[brightest_channel_index] = normalize_brightness(brightest_channel)
    result[darkest_channel_index] = normalize_brightness(darkest_channel)
    return result


def normalize_brightness(channel):
    brightest_value = np.max(channel)
    darkest_value = np.min(channel)
    scale = brightest_value - darkest_value
    return (channel - darkest_value) / scale


# align the images
# functions that might be useful for aligning the images include:
# np.roll, np.sum, sk.transform.rescale (for multiscale)
# ag = g
# ar = r

r_cropped, g_cropped, b_cropped = crop(r, g, b)

skio.imshow_collection([b, b_cropped])
skio.show()

ar = pyramid_align(r_cropped, b_cropped)
ag = pyramid_align(g_cropped, b_cropped)
# create a color image
im_stacked = np.dstack([r, g, b])
im_out = np.dstack([ar, ag, b_cropped])
im_manual = np.dstack([np.roll(np.roll(r, 120, axis=0), 8, axis=1), np.roll(np.roll(g, 50, axis=0), 30, axis=1), b])
# save the image
fname = './out_path/out_fname.jpg'
#skio.imsave(fname, im_out)
print("cropped size: ", r_cropped.shape)
# display the image
skio.imshow(im_out)
skio.imshow_collection([ar, ag, b_cropped])
print(b_cropped[50, :50])
cr, cg, cb = auto_contrast(ar, ag, b_cropped)
print(cb[50, :50])
skio.imshow_collection([np.gradient(r_cropped)[0] * 100, np.gradient(b_cropped)[0] * 100])
skio.show()
skio.imshow_collection([im_stacked, im_out])
skio.imshow_collection([im_out, np.dstack([cr, cg, cb])])
skio.imshow_collection([r_cropped, g_cropped, b_cropped, im_out])
skio.show()
