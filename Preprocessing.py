__author__ = 'gbito'
import numpy
from scipy import misc
from pylearn2.utils import wraps
from pylearn2.utils.rng import make_np_rng
from ift6266h15.code.pylearn2.datasets.variable_image_dataset import BaseImageTransformer

class Preprocess(BaseImageTransformer):
   """
   This crops the image to a square shapes and converts it from RGB to grayscale

   Parameters
   ----------
   scaled_size : int
       Size of the smallest side of the image after rescaling
   crop_size : int
       Size of the square crop. Must be bigger than scaled_size.
   rng : int or rng, optional
       RNG or seed for an RNG
   convert_rbg_to_grayscale: bool
      converts the input image from RGB to grayscale
   """
   _default_seed = 2015 + 1 + 18

   def __init__(self, scaled_size, crop_size, rng=_default_seed, convert_rbg_to_grayscale=False):
      self.scaled_size = scaled_size
      self.crop_size = crop_size
      assert self.scaled_size > self.crop_size
      self.rng = make_np_rng(rng, which_method="random_integers")

      self.saveImgCount = 0
      self.imageMeanSize = 0
      self.imageCounter = 0

   @wraps(BaseImageTransformer.get_shape)
   def get_shape(self):
      return (self.crop_size, self.crop_size)

   @wraps(BaseImageTransformer.preprocess)
   def preprocess(self, image):
      small_axis = numpy.argmin(image.shape[:-1])
      ratio = (1.0 * self.scaled_size) / image.shape[small_axis]
      resized_image = misc.imresize(image[:, :, 0], ratio)

      max_i = resized_image.shape[0] - self.crop_size
      max_j = resized_image.shape[1] - self.crop_size
      i = self.rng.randint(low=0, high=max_i)
      j = self.rng.randint(low=0, high=max_j)

      cropped_image_temp = resized_image[i: i + self.crop_size,
                      j: j + self.crop_size]

      cropped_image = cropped_image_temp[:, :, numpy.newaxis]
      # numpy.empty((cropped_image_temp.shape[0], cropped_image_temp.shape[1], 1))*0
      # cropped_image[:, :, 0] = cropped_image_temp



      # self.imageMeanSize = (1.0*self.imageCounter*self.imageMeanSize + image.shape[small_axis])/(self.imageCounter+1.0)
      # self.imageCounter += 1
      #
      # if self.imageCounter % 500 == 0:
      #    print "the mean size is: ", self.imageMeanSize
      #
      # if self.saveImgCount < 50:
      #
      #    if self.scaled_size < image.shape[small_axis]:
      #       misc.imsave("Test_Images/image_bigger_image_{}.jpg".format(self.saveImgCount), cropped_image)
      #
      #    if self.scaled_size > image.shape[small_axis]:
      #       misc.imsave("Test_Images/image_smaller_image_{}.jpg".format(self.saveImgCount), cropped_image)
      #
      #    self.saveImgCount += 1



      return cropped_image
