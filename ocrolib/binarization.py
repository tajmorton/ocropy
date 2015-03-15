import logging

from pylab import *
from scipy.ndimage import filters, interpolation, morphology, measurements
from scipy import stats

import ocrolib
from .common import OcropusException
from .common import dshow

logger = logging.getLogger("ocropus.ocrolib.binarization")
logger.setLevel(logging.DEBUG)


class BinarizationException(OcropusException):
    pass


class EmptyImageException(BinarizationException):
    pass


class PageSizeException(BinarizationException):
    pass


class InvertedImageException(BinarizationException):
    pass


class BitDepthException(BinarizationException):
    pass


def raise_for_image_error(image):
    if len(image.shape) == 3:
        raise BitDepthException("Input image is color image: {shape}".format(
                                shape=image.shape))

    inverted = amax(image) - image
    if mean(inverted) < median(inverted):
        raise InvertedImageException("Image may be inverted")

    h, w = image.shape

    if h < 600:
        raise PageSizeException("Image not tall enough for a page image: "
                                "{shape}".format(shape=image.shape))
    elif h > 10000:
        raise PageSizeException("Image too tall for a page image: "
                                "{shape}".format(shape=image.shape))
    elif w < 600:
        raise PageSizeException("Image too narrow for a page image "
                                "{shape}".format(shape=image.shape))
    elif w > 10000:
        raise PageSizeException("Line too wide for a page image "
                                "{shape}".format(shape=image.shape))

    return True


def estimate_skew_angle(image, angles, progress_timeout):
    estimates = []
    for a in angles:
        v = mean(
            interpolation.rotate(image, a, order=0, mode='constant'),
            axis=1)
        v = var(v)
        estimates.append((v, a))
    if progress_timeout:
        plot([y for x, y in estimates], [x for x, y in estimates])
        ginput(1, progress_timeout)
    _, a = max(estimates)
    return a


def binarize_image(input_image,
                   output_fname=None,
                   output_raw_fname=None,
                   output_norm_fname=None,
                   progress_timeout=0,
                   validate_image=False,
                   force_grayscale_conversion=False,
                   whitelevel_scale_amount=0.5,
                   whitelevel_percentile=80,
                   whitelevel_range=20,
                   max_skew_angle=2.0,
                   skew_step_angle=8,
                   border_ignore_amount=0.1,
                   text_region_mask_scale=1.0,
                   black_est_percentile=5.0,
                   white_est_percentile=90.0,
                   binarization_threshold=0.5):
    """
    Performs binarization on an input image.

    Parameters:
    * input_image: Grayscale image read with ocrolib.read_image_gray()
    * output_fname: If specified, filename to write binary image to
    * output_raw_fname: If specified, filename to write input image to
    * output_norm_fname: If specified, filename to write normalized image to
    * progress_timeout: If non-zero, number of seconds to display intermediate
        images for. Requires an interactive Matplotlib backend. 
    * validate_image: If True, performs validations on the input image,
        raising BinarizationExceptions on errors.
    * force_grayscale_conversion: If True, performs image binarization
        even if image already appears to be binary.
    * whitelevel_scale_amount: Amount to scale input image by to esimate
        image whitelevel.
    * whitelevel_percentile: Percentile filter value for whitelevel
        estimate.
    * whitelevel_range: Number of neighboring pixels to use for whitelevel
        estimate at each location.
    * max_skew_angle: Maximum amount (in degrees) to rotate image by to adjust
        for skew (in one direction).
    * skew_step_angle: Number of rotations to test per degree of rotation. 
    * border_ignore_amount: Proportion of image (starting from the edges) to
        ignore when estimating skew.
    * black_est_percentile: Percentile for black estimation.
    * white_est_percentile: Percentile for white estimation.
    * binarization_threshold: Threshold for final image binarization after
        normalization.
    """ 
    raw = input_image

    if progress_timeout:
        dshow(raw, "input", progress_timeout)

    # perform image normalization
    image = raw - amin(raw)  # rescale so darkest regions are 0
    if amax(image) == amin(image):
        raise EmptyImageException("Image is empty.")

    # Rescale image so channels are between 0-1
    image /= amax(image)

    if validate_image:
        raise_for_image_error(image)

    # check whether the image is already effectively binarized
    if force_grayscale_conversion:
        extreme = 0
    else:
        # Count of pixels which are very dark or very light
        num_extreme = (sum(image < 0.05) + sum(image > 0.95))

        # Percentage of pixels which are very dark or very light
        extreme = float(num_extreme)/prod(image.shape)

    if extreme > 0.95:
        logger.info("No image normalization performed "
                    "(extreme={extreme}).".format(extreme=extreme))
        flat = image
    else:
        # if not, we need to flatten it by estimating the local whitelevel
        logger.info("Flattening image")
        scaled = interpolation.zoom(image, whitelevel_scale_amount)

        # Apply percentile filter along the x axis (whitelevel_range
        # pixels in X, 2 in Y):
        m = filters.percentile_filter(scaled, whitelevel_percentile,
                                      size=(whitelevel_range, 2))

        # Apply percentile filter to already-filtered image along the y axis:
        m = filters.percentile_filter(m, whitelevel_percentile,
                                      size=(2, whitelevel_range))

        # Rescale white-level back to orig. size:
        m = interpolation.zoom(m, 1.0/whitelevel_scale_amount)
        if progress_timeout:
            clf()
            imshow(m, vmin=0, vmax=1)
            ginput(1, progress_timeout)

        w, h = minimum(array(image.shape), array(m.shape))
        flat = clip(image[:w, :h] - m[:w, :h] + 1, 0, 1)

        if progress_timeout:
            clf()
            imshow(flat, vmin=0, vmax=1)
            ginput(1, progress_timeout)

    # estimate skew angle and rotate
    if max_skew_angle > 0:
        logger.info("Estimating skew angle")
        d0, d1 = flat.shape
        o0, o1 = int(border_ignore_amount*d0), int(border_ignore_amount*d1)

        flat = amax(flat) - flat
        flat -= amin(flat)

        # Flattened image with margin cropped out for skew estimation:
        est = flat[o0:d0-o0, o1:d1-o1]
        ma = max_skew_angle
        # Total number of rotations to test: 
        ms = int(2*max_skew_angle*skew_step_angle)

        angle = estimate_skew_angle(est, linspace(-ma, ma, ms+1),
                                    progress_timeout)
        flat = interpolation.rotate(flat, angle, mode='constant', reshape=0)
        flat = amax(flat) - flat
    else:
        angle = 0

    # estimate low and high thresholds
    logger.info("Estimating image thresholds")
    d0, d1 = flat.shape
    o0, o1 = int(border_ignore_amount*d0), int(border_ignore_amount*d1)
    est = flat[o0:d0-o0, o1:d1-o1]

    if text_region_mask_scale > 0:
        # by default, we use only regions that contain
        # significant variance; this makes the percentile
        # based low and high estimates more reliable
        e = text_region_mask_scale
        v = est - filters.gaussian_filter(est, e*20.0)
        v = filters.gaussian_filter(v**2, e*20.0)**0.5

        v = (v > 0.3 * amax(v))
        v = morphology.binary_dilation(v, structure=ones((e*50, 1)))
        v = morphology.binary_dilation(v, structure=ones((1, e*50)))

        if progress_timeout:
            imshow(v)
            ginput(1, progress_timeout)
        est = est[v]

    lo = stats.scoreatpercentile(est.ravel(), black_est_percentile)
    hi = stats.scoreatpercentile(est.ravel(), white_est_percentile)

    # rescale the image to get the gray scale image
    logger.info("Rescaling image")
    flat -= lo
    flat /= (hi-lo)
    flat = clip(flat, 0, 1)
    if progress_timeout:
        imshow(flat, vmin=0, vmax=1)
        ginput(1, progress_timeout)

    bin = 1 * (flat > binarization_threshold)

    # output the normalized grayscale and the thresholded images
    logger.info("Normalization complete. Low={low}, high={high}, "
                "angle={angle}.".format(low=lo, high=hi, angle=angle))

    logger.info("Writing image")
    if progress_timeout:
        clf()
        gray()
        imshow(bin)
        ginput(1, progress_timeout)

    if output_fname:
        ocrolib.write_image_binary(output_fname, bin)

    if output_raw_fname:
        ocrolib.write_image_gray(output_raw_fname, raw)

    if output_norm_fname:
        ocrolib.write_image_gray(output_norm_fname, flat)

    return bin



