from PIL import Image
import math

# get N_skin and N_background for the training image
# map pixel indices to either 0(background) or 1(skin)
dictionary = {}
N_skin = 0
N_background = 0
im = Image.open('family.png', 'r')  # read ground-truth
pix_value = list(im.getdata())  # extract (R,G,B,A) values to a list
for idx, pixel in enumerate(pix_value):  # because white(255,255,255) and black(0,0,0)
    if pixel[0] == 255:
        N_skin += 1
        dictionary[idx] = 1
    else:
        N_background += 1
        dictionary[idx] = 0
print('N_skin for training image is', N_skin)
print('N_background for training image is', N_background)
# convert pixel values to rg Chroma Space
r_list_background = []
g_list_background = []
r_list_skin = []
g_list_skin = []
im = Image.open('family.jpg', 'r')
pix_value = list(im.getdata())  # extract (R,G,B,A) values to a list
for idx, pix in enumerate(pix_value):
    RGB_sum = pix[0] + pix[1] + pix[2]
    if dictionary[idx] == 0:  # belongs to background
        if RGB_sum == 0:
            R = G = 0
        else:
            R = pix[0] / RGB_sum  # rk = Rk / (Rk+Gk+Bk)
            G = pix[1] / RGB_sum  # gk = Gk / (Rk+Gk+Bk)
        r_list_background.append(R)
        g_list_background.append(G)
    elif dictionary[idx] == 1:  # belongs to skin
        if RGB_sum == 0:
            R = G = 0
        else:
            R = pix[0] / RGB_sum  # rk = Rk / (Rk+Gk+Bk)
            G = pix[1] / RGB_sum  # gk = Gk / (Rk+Gk+Bk)
        r_list_skin.append(R)
        g_list_skin.append(G)
print('the length for r_list_skin is', len(r_list_skin))
print('the length for r_list_background is', len(r_list_background))
print('rg Chroma Space processed')

# call background pixels class 0 and skin pixels class 1, get the parameters for mean and variance
mean_0r = sum(r_list_background) / N_background
diff_0r = [(ri - mean_0r) ** 2 for ri in r_list_background]
variance_0r = sum(diff_0r) / N_background

mean_0g = sum(g_list_background) / N_background
diff_0g = [(gi - mean_0g) ** 2 for gi in g_list_background]
variance_0g = sum(diff_0g) / N_background


mean_1r = sum(r_list_skin) / N_skin
diff_1r = [(ri - mean_1r) ** 2 for ri in r_list_skin]
variance_1r = sum(diff_1r) / N_skin

mean_1g = sum(g_list_skin) / N_skin
diff_1g = [(gi - mean_1g) ** 2 for gi in g_list_skin]
variance_1g = sum(diff_1g) / N_skin
print('train parameters done!')

# Testing Stage
# get P(background) and P(skin) for the test image
N_skin = 0
N_background = 0
img = Image.open('portrait.png', 'r')  # read ground-truth image
pix_value = list(img.getdata())  # extract (R,G,B,A) values to a list
for pixel in pix_value:  # because white(255,255,255) and black(0,0,0)
    if pixel[0] == 255:
        N_skin += 1
    else:
        N_background += 1
print('N_skin for test image is', N_skin)
print('N_background for test image is', N_background)
Pb = N_background / (N_background + N_skin)
Ps = N_skin / (N_background + N_skin)
print('the probability of being background pixels is', Pb)
print('the probability of being skin pixels is', Ps)


# calculate the likelihood of data x under Gaussian Distribution
def get_cond_probability(mean, value, variance):
    return 1.0 / (math.pow((2 * math.pi), 1 / 2) * variance ** (1 / 2)) * math.exp(
        -1 / 2 * (value - mean) ** 2 / variance)


# get meta-data for testing image
image = Image.open('portrait.jpg', 'r')  # read testing image
width, height = image.size
pixel_test = image.load()


# Apply Bayesian Decision Rule to classify pixels and create binary-mask image
def create_image():
    for i in range(width):
        for j in range(height):
            r, g, b = image.getpixel((i, j))
            rgb_sum = r + g + b
            if rgb_sum == 0:
                rk = gk = 0
            else:
                rk = r / rgb_sum
                gk = g / rgb_sum
            # joint probability P(x | H0)
            p_x_background = get_cond_probability(mean_0r, rk, variance_0r) * get_cond_probability(mean_0g, gk, variance_0g)
            # joint probability of P(x | H1)
            p_x_skin = get_cond_probability(mean_1r, rk, variance_1r) * get_cond_probability(mean_1g, gk, variance_1g)
            if p_x_skin / p_x_background >= Pb / Ps:  # Bayesian Decision Rule
                pixel_test[i, j] = (255, 255, 255)  # classify the pixel as skin(white)
            else:
                pixel_test[i, j] = (0, 0, 0)  # classify the pixel as background(black)
    image.save('binary_mask.png')  # save the binary mask image
    print('image saved!')


# calculate the true positive rate, true negative rate
# false positive rate, as well as false negative rate
def get_rates():
    true_positive = true_negative = false_positive = false_negative = 0
    wid, hei = img.size  # ground truth image
    pixel_truth = img.load()
    for i in range(wid):
        for j in range(hei):
            if pixel_truth[i, j][0] == 255 and pixel_test[i, j][0] == 255:
                true_positive += 1  # true skin pixels that are classified as skin
            elif pixel_truth[i, j][0] == 255 and pixel_test[i, j][0] == 0:
                false_negative += 1  # skin pixels that are classified as background
            elif pixel_truth[i, j][0] == 0 and pixel_test[i, j][0] == 0:
                true_negative += 1   # background pixels that are classified as background
            elif pixel_truth[i, j][0] == 0 and pixel_test[i, j][0] == 255:
                false_positive += 1  # backgrounds pixels that are classified as skin
    true_positive = true_positive / N_skin
    print('True Positive Rate is', true_positive)
    true_negative = true_negative / N_background
    print('True Negative Rate is', true_negative)
    false_positive = false_positive / N_background
    print('False Positive Rate is', false_positive)
    false_negative = false_negative / N_skin
    print('False Negative Rate is', false_negative)


# append original image, ground-truth image and binary mask image for comparisons
def append_image():
    images = [Image.open(x) for x in ['portrait.jpg', 'portrait.png', 'binary_mask.png']]
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_ima = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for ima in images:
        new_ima.paste(ima, (x_offset, 0))
        x_offset += ima.size[0]
    new_ima.save('comparisons.png')


def main():
    create_image()
    get_rates()
    append_image()


main()


