__author__ = 'Daniel E-B'
import numpy as np
import scipy.misc as misc
from ift6266h15.code.pylearn2.datasets.variable_image_dataset import BaseImageTransformer

class CropImage(BaseImageTransformer):
      """

      """
      def __init__(self, scaled_size, crop_size):
         self.scaled_size = scaled_size
         self.crop_size = crop_size

         assert self.scaled_size > self.crop_size

      @wraps(BaseImageTransformer.get_shape)
      def get_shape(self):
         return (self.crop_size, self.crop_size)

      @wraps(BaseImageTransformer.preprocess)
      def preprocess(self, image):
         """
         This function crops an image to the size specified by crop_size
         :param image:
         :return:
         """
         small_axis = np.argmin(image.shape[:-1]) # pick the smallest dimension
         ratio = (1.0 * self.scaled_size) / image.shape[small_axis]
         resized_image = misc.imresize(image, ratio)

         max_horizontal_delta = resized_image.shape[0] - self.crop_size
         max_vertical_delta   = resized_image.shape[1] - self.crop_size

         horizontal_delta  = self.rng.randint(low=0, high=max_horizontal_delta)
         vertical_delta    = self.rng.randint(low=0, high=max_vertical_delta)

         cropped_image = resized_image[horizontal_delta:horizontal_delta+self.crop_size,
                                       vertical_delta:vertical_delta+self.crop_size, :]

         return cropped_image
