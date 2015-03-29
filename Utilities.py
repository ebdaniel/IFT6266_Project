__author__ = 'Daniel E-B'
import numpy as np
import scipy.misc as misc
import ConvNet as CNN

from os import system as sys
import pylearn2.scripts.plot_monitor as PM
from ift6266h15.code.pylearn2.datasets.variable_image_dataset import BaseImageTransformer

# class CropImage(BaseImageTransformer):
#       """
#
#       """
#       def __init__(self, scaled_size, crop_size):
#          self.scaled_size = scaled_size
#          self.crop_size = crop_size
#
#          assert self.scaled_size > self.crop_size
#
#       @wraps(BaseImageTransformer.get_shape)
#       def get_shape(self):
#          return (self.crop_size, self.crop_size)
#
#       @wraps(BaseImageTransformer.preprocess)
#       def preprocess(self, image):
#          """
#          This function crops an image to the size specified by crop_size
#          :param image:
#          :return:
#          """
#          small_axis = np.argmin(image.shape[:-1]) # pick the smallest dimension
#          ratio = (1.0 * self.scaled_size) / image.shape[small_axis]
#          resized_image = misc.imresize(image, ratio)
#
#          max_horizontal_delta = resized_image.shape[0] - self.crop_size
#          max_vertical_delta   = resized_image.shape[1] - self.crop_size
#
#          horizontal_delta  = self.rng.randint(low=0, high=max_horizontal_delta)
#          vertical_delta    = self.rng.randint(low=0, high=max_vertical_delta)
#
#          cropped_image = resized_image[horizontal_delta:horizontal_delta+self.crop_size,
#                                        vertical_delta:vertical_delta+self.crop_size, :]
#
#          return cropped_image


def find_hyper_parameters():
   # Create dataset
   crop_size = 224
   train, valid, test = CNN.build_dogs_vs_cats_dataset(crop_size)

   # Create model
   # TODO: these tests should be ran automatically

   # Build a dictionnary of different tests to run
   kernel_shapes = [[[5, 5], [5, 5], [4, 4], [4, 4], [4, 4]]]

   output_channels_convnet = [[32, 32, 128, 128, 32]]

   output_channels_fullnet = [[128, 128]]

   conv_layers_tests = []
   full_layers_tests = []
   conv_test_idx = 0

   # create conv layers tests
   for i in xrange(len(kernel_shapes)):  # This loops on the dimension of the first layer convnet

      for out_channels_idx in range(len(output_channels_convnet)):
         conv_layers_tests.extend([{'nb_layers': 5,
                                    'output_channels': output_channels_convnet[out_channels_idx],
                                    'kernel_shape': kernel_shapes[i],
                                    'kernel_stride': [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
                                    'pool_shape': [[2, 2], [2, 2], None, None, [2, 2]],
                                    'pool_stride': [[2, 2], [2, 2], None, None, [2, 2]],
                                    'weight_decay': [.0001, .0001, .0001, .0001, .0001],
                                    'pool_type': ['max', 'max', None, None, 'max']}])

         conv_test_idx += 1



   # create fully connected layers tests
   for out_channels_idx in range(len(output_channels_fullnet)):
      full_layers_tests.extend([{'nb_layers': 2,
                                 'dim': output_channels_fullnet[out_channels_idx],
                                 'weight_decay': [.0001, .0001]}])

   print "Total number of tests: ", len(conv_layers_tests) * len(full_layers_tests)

   # run the tests with different configurations
   # list of good answers
   # test_idx = 0
   # for i in range(len(conv_layers_tests)):
   #    for j in range(len(full_layers_tests)):
   #       test_idx += 1
   #       test_best_result = 'test_{}_best_result.pkl'.format(test_idx)
   #       test_results = 'test_{}_results.pkl'.format(test_idx)
   #       #
   #       # print '{}: {}: {}: {}'.format(test_name, conv_layers_tests[i]['output_channels'],
   #       #                           conv_layers_tests[i]['kernel_shape'],
   #       #                           full_layers_tests[j]['dim'])
   #
   #       # index = (len(full_layers_tests)*i)+(j+1)
   #       #
   #       # if test_idx in best_results:
   #       #    print '{}: {}: {}: {}'.format(test_name, conv_layers_tests[i]['output_channels'], conv_layers_tests[i]['kernel_shape'],
   #       #                            full_layers_tests[j]['dim'])
   #
   #       # Build model
   #       model = CNN.build_model(train=train,
   #                               validation=valid,
   #                               test=test,
   #                               crop_size=crop_size,
   #                               conv_layers=conv_layers_tests[i],
   #                               fully_connected_layers=full_layers_tests[j],
   #                               use_weight_decay=True,
   #                               use_drop_out=False,
   #                               batch_size=50,
   #                               best_result_file=test_best_result,
   #                               results_file=test_results,
   #                               monitor_results=True)
   #
   #       try:
   #          # Run model
   #          CNN.run(model)
   #       except (KeyboardInterrupt, SystemExit):
   #          raise
   #       except:
   #          print "Smells fishy! But continue anyways"


   momentum = [{'initial_value': 0.5, 'start': 1, 'saturate': 20, 'final_value': 0.99},
               {'initial_value': 0.5, 'start': 10, 'saturate': 20, 'final_value': 0.99},
               {'initial_value': 0.8, 'start': 1, 'saturate': 10, 'final_value': 0.99}]

   learning_rate = [{'initial_value': 0.01, 'start': 30, 'saturate': 50, 'decay_factor': 0.01},
                    {'initial_value': 0.01, 'start': 20, 'saturate': 50, 'decay_factor': 0.01},
                    {'initial_value': 0.01, 'start': 50, 'saturate': 100, 'decay_factor': 0.01}]

   test_idx = 0
   for i in range(len(momentum)):
      for j in range(len(learning_rate)):

         test_idx += 1
         test_best_result = 'test_{}_best_result.pkl'.format(test_idx)
         test_results = 'test_{}_results.pkl'.format(test_idx)

         # Build model
         model = CNN.build_model(train=train,
                                 validation=valid,
                                 test=test,
                                 crop_size=crop_size,
                                 conv_layers=conv_layers_tests[0],
                                 fully_connected_layers=full_layers_tests[0],
                                 use_weight_decay=True,
                                 use_drop_out=False,
                                 batch_size=50,
                                 best_result_file=test_best_result,
                                 results_file=test_results,
                                 monitor_results=True,
                                 momentum=momentum[i],
                                 learning_rate=learning_rate[j])

         try:
            # Run model
            CNN.run(model)
         except (KeyboardInterrupt, SystemExit):
            raise
         except:
            print "Smells fishy! But continue anyways"



def plot_results(file_path):
   commande = "{} {}".format(PM, file_path)
   sys(commande)