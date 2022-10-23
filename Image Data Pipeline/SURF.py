
import numpy as np
from cv2 import resize, GaussianBlur, subtract, KeyPoint, INTER_LINEAR, INTER_NEAREST
from functools import cmp_to_key
import logging
import cv2
import matplotlib.pyplot as plt

class SURF:
    def __init__(self):
        pass

    def generateBaseImage(self, image, sigma, assumed_blur):
        """Generate base image from input image by upsampling by 2 in both directions and blurring
        """
        image = resize(image, (0, 0), fx=2, fy=2, interpolation=INTER_LINEAR)
        sigma_diff = np.sqrt(max((sigma ** 2) - ((2 * assumed_blur) ** 2), 0.01))
        return GaussianBlur(image, (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff)  # the image blur is now sigma instead of assumed_blur

    def computeNumberOfOctaves(self, image_shape):
        """Compute number of octaves in image pyramid as function of base image shape (OpenCV default)"""
        return int(round(np.log(min(image_shape)) / np.log(2) - 1))

    def generateGaussianKernels(self, sigma, num_intervals):
        """Generate list of gaussian kernels at which to blur the input image. Default values of sigma, intervals, and octaves follow section 3 of Lowe's paper.
        """
        num_images_per_octave = num_intervals + 3
        k = 2 ** (1. / num_intervals)
        gaussian_kernels = np.zeros(num_images_per_octave)  # scale of gaussian blur necessary to go from one blur scale to the next within an octave
        gaussian_kernels[0] = sigma

        for image_index in range(1, num_images_per_octave):
            sigma_previous = (k ** (image_index - 1)) * sigma
            sigma_total = k * sigma_previous
            gaussian_kernels[image_index] = np.sqrt(sigma_total ** 2 - sigma_previous ** 2)
        return gaussian_kernels

    def generateGaussianImages(self, image, num_octaves, gaussian_kernels):
        """Generate scale-space pyramid of Gaussian images
        """
        gaussian_images = []

        for octave_index in range(num_octaves):
            gaussian_images_in_octave = []
            gaussian_images_in_octave.append(image)  # first image in octave already has the correct blur
            for gaussian_kernel in gaussian_kernels[1:]:
                image = GaussianBlur(image, (0, 0), sigmaX=gaussian_kernel, sigmaY=gaussian_kernel)
                gaussian_images_in_octave.append(image)
            gaussian_images.append(gaussian_images_in_octave)
            octave_base = gaussian_images_in_octave[-3]
            image = resize(octave_base, (int(octave_base.shape[1] / 2), int(octave_base.shape[0] / 2)), interpolation=INTER_NEAREST)
        return np.array(gaussian_images)

    def generateDoGImages(self, gaussian_images):
        """Generate Difference-of-Gaussians image pyramid
        """
        dog_images = []

        for gaussian_images_in_octave in gaussian_images:
            dog_images_in_octave = []
            for first_image, second_image in zip(gaussian_images_in_octave, gaussian_images_in_octave[1:]):
                dog_images_in_octave.append(subtract(second_image, first_image))  # ordinary subtraction will not work because the images are unsigned integers
            dog_images.append(dog_images_in_octave)
        return np.array(dog_images)

    def findScaleSpaceExtrema(self, gaussian_images, dog_images, num_intervals, sigma, image_border_width, contrast_threshold=0.04):
        """Find pixel positions of all scale-space extrema in the image pyramid
        """
        threshold = np.floor(0.5 * contrast_threshold / num_intervals * 255)  # from OpenCV implementation
        keypoints = []

        for octave_index, dog_images_in_octave in enumerate(dog_images):
            for image_index, (first_image, second_image, third_image) in enumerate(zip(dog_images_in_octave, dog_images_in_octave[1:], dog_images_in_octave[2:])):
                # (i, j) is the center of the 3x3 array
                for i in range(image_border_width, first_image.shape[0] - image_border_width):
                    for j in range(image_border_width, first_image.shape[1] - image_border_width):
                        if self.isPixelAnExtremum(first_image[i-1:i+2, j-1:j+2], second_image[i-1:i+2, j-1:j+2], third_image[i-1:i+2, j-1:j+2], threshold):
                            localization_result = self.localizeExtremumViaQuadraticFit(i, j, image_index + 1, octave_index, num_intervals, dog_images_in_octave, sigma, contrast_threshold, image_border_width)
                            if localization_result is not None:
                                keypoint, localized_image_index = localization_result
                                keypoints_with_orientations = self.computeKeypointsWithOrientations(keypoint, octave_index, gaussian_images[octave_index][localized_image_index])
                                for keypoint_with_orientation in keypoints_with_orientations:
                                    keypoints.append(keypoint_with_orientation)
        return keypoints

    def isPixelAnExtremum(self, first_subimage, second_subimage, third_subimage, threshold):
        """Return True if the center element of the 3x3x3 input array is strictly greater than or less than all its neighbors, False otherwise
        """
        center_pixel_value = second_subimage[1, 1]
        print(center_pixel_value)
        if abs(center_pixel_value) > threshold:
            if center_pixel_value > 0:
                return all(center_pixel_value >= first_subimage) and \
                    all(center_pixel_value >= third_subimage) and \
                    all(center_pixel_value >= second_subimage[0, :]) and \
                    all(center_pixel_value >= second_subimage[2, :]) and \
                    center_pixel_value >= second_subimage[1, 0] and \
                    center_pixel_value >= second_subimage[1, 2]
            elif center_pixel_value < 0:
                return all(center_pixel_value <= first_subimage) and \
                    all(center_pixel_value <= third_subimage) and \
                    all(center_pixel_value <= second_subimage[0, :]) and \
                    all(center_pixel_value <= second_subimage[2, :]) and \
                    center_pixel_value <= second_subimage[1, 0] and \
                    center_pixel_value <= second_subimage[1, 2]
        return False

    def localizeExtremumViaQuadraticFit(self, i, j, image_index, octave_index, num_intervals, dog_images_in_octave, sigma, contrast_threshold, image_border_width, eigenvalue_ratio=10, num_attempts_until_convergence=5):
        """Iteratively refine pixel positions of scale-space extrema via quadratic fit around each extremum's neighbors
        """
        extremum_is_outside_image = False
        image_shape = dog_images_in_octave[0].shape
        for attempt_index in range(num_attempts_until_convergence):
            # need to convert from uint8 to float32 to compute derivatives and need to rescale pixel values to [0, 1] to apply Lowe's thresholds
            first_image, second_image, third_image = dog_images_in_octave[image_index-1:image_index+2]
            pixel_cube = np.stack([first_image[i-1:i+2, j-1:j+2],
                                second_image[i-1:i+2, j-1:j+2],
                                third_image[i-1:i+2, j-1:j+2]]).astype('float32') / 255.
            gradient = self.computeGradientAtCenterPixel(pixel_cube)
            hessian = self.computeHessianAtCenterPixel(pixel_cube)
            extremum_update = -np.linalg.lstsq(hessian, gradient, rcond=None)[0]
            if abs(extremum_update[0]) < 0.5 and abs(extremum_update[1]) < 0.5 and abs(extremum_update[2]) < 0.5:
                break
            j += int(round(extremum_update[0]))
            i += int(round(extremum_update[1]))
            image_index += int(round(extremum_update[2]))
            # make sure the new pixel_cube will lie entirely within the image
            if i < image_border_width or i >= image_shape[0] - image_border_width or j < image_border_width or j >= image_shape[1] - image_border_width or image_index < 1 or image_index > num_intervals:
                extremum_is_outside_image = True
                break
        if extremum_is_outside_image:
            return None
        if attempt_index >= num_attempts_until_convergence - 1:
            return None
        functionValueAtUpdatedExtremum = pixel_cube[1, 1, 1] + 0.5 * np.dot(gradient, extremum_update)
        if abs(functionValueAtUpdatedExtremum) * num_intervals >= contrast_threshold:
            xy_hessian = hessian[:2, :2]
            xy_hessian_trace = np.trace(xy_hessian)
            xy_hessian_det = np.linalg.det(xy_hessian)
            if xy_hessian_det > 0 and eigenvalue_ratio * (xy_hessian_trace ** 2) < ((eigenvalue_ratio + 1) ** 2) * xy_hessian_det:
                # Contrast check passed -- construct and return OpenCV KeyPoint object
                keypoint = KeyPoint()
                keypoint.pt = ((j + extremum_update[0]) * (2 ** octave_index), (i + extremum_update[1]) * (2 ** octave_index))
                keypoint.octave = octave_index + image_index * (2 ** 8) + int(round((extremum_update[2] + 0.5) * 255)) * (2 ** 16)
                keypoint.size = sigma * (2 ** ((image_index + extremum_update[2]) / np.float32(num_intervals))) * (2 ** (octave_index + 1))  # octave_index + 1 because the input image was doubled
                keypoint.response = abs(functionValueAtUpdatedExtremum)
                return keypoint, image_index
        return None

    def computeGradientAtCenterPixel(pixel_array):
        """Approximate gradient at center pixel [1, 1, 1] of 3x3x3 array using central difference formula of order O(h^2), where h is the step size
        """
        # With step size h, the central difference formula of order O(h^2) for f'(x) is (f(x + h) - f(x - h)) / (2 * h)
        # Here h = 1, so the formula simplifies to f'(x) = (f(x + 1) - f(x - 1)) / 2
        # NOTE: x corresponds to second array axis, y corresponds to first array axis, and s (scale) corresponds to third array axis
        dx = 0.5 * (pixel_array[1, 1, 2] - pixel_array[1, 1, 0])
        dy = 0.5 * (pixel_array[1, 2, 1] - pixel_array[1, 0, 1])
        ds = 0.5 * (pixel_array[2, 1, 1] - pixel_array[0, 1, 1])
        return np.array([dx, dy, ds])

    def computeHessianAtCenterPixel(pixel_array):
        """Approximate Hessian at center pixel [1, 1, 1] of 3x3x3 array using central difference formula of order O(h^2), where h is the step size
        """
        # With step size h, the central difference formula of order O(h^2) for f''(x) is (f(x + h) - 2 * f(x) + f(x - h)) / (h ^ 2)
        # Here h = 1, so the formula simplifies to f''(x) = f(x + 1) - 2 * f(x) + f(x - 1)
        # With step size h, the central difference formula of order O(h^2) for (d^2) f(x, y) / (dx dy) = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h ^ 2)
        # Here h = 1, so the formula simplifies to (d^2) f(x, y) / (dx dy) = (f(x + 1, y + 1) - f(x + 1, y - 1) - f(x - 1, y + 1) + f(x - 1, y - 1)) / 4
        # NOTE: x corresponds to second array axis, y corresponds to first array axis, and s (scale) corresponds to third array axis
        center_pixel_value = pixel_array[1, 1, 1]
        dxx = pixel_array[1, 1, 2] - 2 * center_pixel_value + pixel_array[1, 1, 0]
        dyy = pixel_array[1, 2, 1] - 2 * center_pixel_value + pixel_array[1, 0, 1]
        dss = pixel_array[2, 1, 1] - 2 * center_pixel_value + pixel_array[0, 1, 1]
        dxy = 0.25 * (pixel_array[1, 2, 2] - pixel_array[1, 2, 0] - pixel_array[1, 0, 2] + pixel_array[1, 0, 0])
        dxs = 0.25 * (pixel_array[2, 1, 2] - pixel_array[2, 1, 0] - pixel_array[0, 1, 2] + pixel_array[0, 1, 0])
        dys = 0.25 * (pixel_array[2, 2, 1] - pixel_array[2, 0, 1] - pixel_array[0, 2, 1] + pixel_array[0, 0, 1])
        return np.array([[dxx, dxy, dxs], 
                    [dxy, dyy, dys],
                    [dxs, dys, dss]])

    def computeKeypointsAndDescriptors(self, image, sigma=1.6, num_intervals=3, assumed_blur=0.5, image_border_width=5):
        """Compute SIFT keypoints and descriptors for an input image
        """
        base_image = self.generateBaseImage(image, sigma, assumed_blur)
        cv2.imshow("Base image", base_image)
        cv2.waitKey(0) 
        num_octaves = self.computeNumberOfOctaves(base_image.shape)
        print(f"Number of octaves: {num_octaves}")
        gaussian_kernels = self.generateGaussianKernels(sigma, num_intervals)
        print(f"gaussian kernels: {gaussian_kernels}")
        gaussian_images = self.generateGaussianImages(base_image, num_octaves, gaussian_kernels)
        for image in gaussian_images[0]:
            cv2.imshow("Gaussian image", image)
            cv2.waitKey(0) 
        dog_images = self.generateDoGImages(gaussian_images)
        for image in dog_images[0]:
            cv2.imshow("dog_images", image)
            cv2.waitKey(0) 
        keypoints = self.findScaleSpaceExtrema(gaussian_images, dog_images, num_intervals, sigma, image_border_width)
        print(keypoints)
        # keypoints = self.removeDuplicateKeypoints(keypoints)
        # keypoints = self.convertKeypointsToInputImageSize(keypoints)
        # descriptors = self.generateDescriptors(keypoints, gaussian_images)
        return keypoints

    def extract_features(self, img):
        keypoints, descriptors = self.computeKeypointsAndDescriptors(img)

        # train_keypoints, train_descriptor = surf.detectAndCompute(training_gray, None)
        # test_keypoints, test_descriptor = surf.detectAndCompute(test_gray, None)

        # keypoints_without_size = np.copy(training_image)
        # keypoints_with_size = np.copy(training_image)

        # cv2.drawKeypoints(training_image, train_keypoints, keypoints_without_size, color = (0, 255, 0))

        # cv2.drawKeypoints(training_image, train_keypoints, keypoints_with_size, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # # Display image with and without keypoints size
        # fx, plots = plt.subplots(1, 2, figsize=(20,10))

        # plots[0].set_title("Train keypoints With Size")
        # plots[0].imshow(keypoints_with_size, cmap='gray')

        # plots[1].set_title("Train keypoints Without Size")
        # plots[1].imshow(keypoints_without_size, cmap='gray')

        # # Print the number of keypoints detected in the training image
        # print("Number of Keypoints Detected In The Training Image: ", len(train_keypoints))

        # # Print the number of keypoints detected in the query image
        # print("Number of Keypoints Detected In The Query Image: ", len(test_keypoints))

        
    def describe_features(self):
        pass