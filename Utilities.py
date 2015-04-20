__author__ = 'Daniel E-B'
import numpy as np
import numpy.random
import scipy.misc as misc
import ConvNet as CNN
import math

from os import system as sys
import pylearn2.scripts.plot_monitor as PM
from ift6266h15.code.pylearn2.datasets.variable_image_dataset import BaseImageTransformer


def find_hyper_parameters():
   # Create dataset
   # crop_size = 125
   # train, valid, test = CNN.build_dogs_vs_cats_dataset(crop_size)

   # Create model
   # TODO: these tests should be ran automatically

   # Build a dictionnary of different tests to run
   kernel_shapes = [[[4, 4], [4, 4], [4, 4], [3, 3], [3, 3], [3, 3]]]

   # kernel_shapes = [[[4, 4], [4, 4], [4, 4], [3, 3], [3, 3], [3, 3]],
   #                  [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]],
   #                  [[4, 4], [4, 4], [3, 3], [3, 3], [3, 3], [3, 3]],
   #                  [[4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4]]]

   output_channels_convnet = [[100, 64, 32, 32, 25, 25]]

   # output_channels_convnet = [[100, 64, 32, 25, 25, 25],
   #                            [64, 64, 32, 32, 32, 32],
   #                            [64, 64, 128, 128, 32, 32]]

   # output_channels_convnet = [[32, 32, 32, 32, 32]]

   output_channels_fullnet = [[32, 32]]

   conv_layers_tests = []
   full_layers_tests = []
   conv_test_idx = 0

   # create conv layers tests
   for i in xrange(len(kernel_shapes)):  # This loops on the dimension of the first layer convnet

      for out_channels_idx in range(len(output_channels_convnet)):
         conv_layers_tests.extend([{'nb_layers': 6,
                                    'output_channels': output_channels_convnet[out_channels_idx],
                                    'kernel_shape': kernel_shapes[i],
                                    'kernel_stride': [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
                                    'pool_shape': [[2, 2], [2, 2], [2, 2], None, None, [2, 2]],
                                    'pool_stride': [[2, 2], [2, 2], [2, 2], None, None, [2, 2]],
                                    'weight_decay': [.0001, .0001, .0001, .0001, .0001, .0001],
                                    'pool_type': ['max', 'max', 'max', None, None, 'max']}])

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


   momentum = [{'initial_value': 0.1, 'start': 1, 'saturate': 20, 'final_value': 0.99}]

   learning_rate = [{'initial_value': 0.1, 'start': 1, 'saturate': 20, 'decay_factor': 0.01}]

   crop_size = [200, 125, 80]

   # train, valid, test = CNN.build_dogs_vs_cats_dataset(crop_size[0])

   print "number of tests: %d"%(len(conv_layers_tests)*len(full_layers_tests)*len(momentum)*len(learning_rate))

   # print "conv_layers_tests: %d"%len(conv_layers_tests)
   # print "full_layers_tests: %d"%len(full_layers_tests)
   # print "momentum: %d"%len(momentum)
   # print "learning_rate: %d"%len(learning_rate)

   test_idx = 19
   for i in range(len(conv_layers_tests)):
      for j in range(len(full_layers_tests)):
         for k in range(len(momentum)):
            for l in range(len(learning_rate)):
               for c in range(len(crop_size)):

                  train, valid, test = CNN.build_dogs_vs_cats_dataset(crop_size[c])

                  test_idx += 1
                  test_best_result = 'test_{}_best_result.pkl'.format(test_idx)
                  test_results = 'test_{}_results.pkl'.format(test_idx)

                  # print "test:{} is conv_layers_tests[output_channels]:{}, conv_layers_tests[kernel_shape]:{}" \
                  #       ", full_layers_tests:{}, momentum:{}, learning_rate:{}".format(test_idx,
                  #                                                                    conv_layers_tests[i]['output_channels'],
                  #                                                                    conv_layers_tests[i]['kernel_shape'],
                  #                                                                    full_layers_tests[j]['dim'],
                  #                                                                    momentum[k],
                  #                                                                    learning_rate[l])

                  # Build model
                  model = CNN.build_model(train=train,
                                          validation=valid,
                                          test=test,
                                          crop_size=crop_size[c],
                                          conv_layers=conv_layers_tests[i],
                                          fully_connected_layers=full_layers_tests[j],
                                          use_weight_decay=True,
                                          use_drop_out=False,
                                          batch_size=50,
                                          best_result_file=test_best_result,
                                          results_file=test_results,
                                          monitor_results=True,
                                          momentum=momentum[k],
                                          learning_rate=learning_rate[l])

                  try:
                     # Run model
                     CNN.run(model)
                  except (KeyboardInterrupt, SystemExit):
                     raise
                  except:
                     print "Smells fishy! But continue anyways"


full_src_image_path  ="/Users/gbito/Documents/Coding/Datasets/dogs_vs_cats/train/"
full_dest_image_path ="/Users/gbito/Documents/Coding/Datasets/dogs_vs_cats/train_ZCA/"


def scale_images(rng, image, scaled_size, crop_size):
   small_axis = np.argmin(image.shape[:-1])
   ratio = (1.0 * scaled_size) / image.shape[small_axis]
   resized_image = misc.imresize(image, ratio)

   max_i = resized_image.shape[0] - crop_size
   max_j = resized_image.shape[1] - crop_size
   i = rng.randint(low=0, high=max_i)
   j = rng.randint(low=0, high=max_j)

   cropped_image = resized_image[i: i + crop_size,
                   j: j + crop_size, :]

   return cropped_image


def crop_whiten_and_save():
   # create a randomnumber generator
   rng = numpy.random.RandomState([1234])

   scale_size  = 230
   crop_size   = 225
   # array containing all the images
   nb_images = 25000
   array_of_images = np.empty((nb_images, crop_size*crop_size, 3))

   print "... scaling"
   for i in range(nb_images/2):
      for j in range(2):
         if j==0:
             # load cat image
            image_name = "cat.{}.jpg".format(i)

         else:
            # load dog image
            image_name = "dog.{}.jpg".format(i)

         # load image file
         image_file_path = full_src_image_path+image_name

         image_file = np.asarray(misc.imread(image_file_path))

         # scale image
         scaled_image = scale_images(rng, image_file, scale_size, crop_size)

         # add scaled image to list
         array_of_images[2*i + j, :, :] = np.reshape(scaled_image, (crop_size*crop_size,3))

   print "... centering"
   # compute centered images
   centered_images = array_of_images - np.mean(array_of_images)

   #
   # assert np.isnan(centered_images).any() == False
   # assert np.isinf(centered_images).any() == False

   print "... whitening"
   for image_idx in range(nb_images):

      # whitening on image patches due to memory
      patch_size = 25
      nb_patches = crop_size*crop_size/patch_size
      withened_images = np.empty((crop_size*crop_size, 3))
      for i in range(nb_patches):
         patch = centered_images[image_idx, patch_size*i:patch_size*(i+1), :]

         for j in range(3):

            # cov_patch = np.cov(patch.T)

            # # compute eigen values and eigen vectors
            # eigvalTemp, eigvecTemp = np.linalg.eigh(cov_patch)
            #
            # # withened image
            # order = np.argsort(-eigvalTemp)
            # eigval = eigvalTemp[order]
            # eigvec = eigvecTemp[:, order]

            # withened_images[:, patch_size*i:patch_size*(i+1)] = np.dot(np.dot(np.dot(eigvec, np.power(eigval, -0.5)), eigvec.T), patch)

            U, s, Vt = np.linalg.svd(a=np.matrix(patch[:, j]), full_matrices=False)

            # U and Vt are the singular matrices, and s contains the singular values.
            # Since the rows of both U and Vt are orthonormal vectors, then U * Vt
            # will be white
            withened_images[patch_size * i:patch_size * (i + 1), j] = np.dot(U, Vt)

      # save withened images to disk
      if image_idx % 2 == 0:
         # save cat image
         image_name = "cat.{}.jpg".format(image_idx/2)
      else:
         # save dog image
         image_name = "dog.{}.jpg".format(image_idx/2)


      print "... saving", image_name


      misc.imsave(full_dest_image_path + image_name, np.reshape(withened_images, (crop_size, crop_size, 3)))


def explore_optimization():
   # Create dataset
   crop_size = 125
   train, valid, test = CNN.build_dogs_vs_cats_dataset(crop_size=crop_size)

   # Create model
   convlayers = {'nb_layers': 6,
                 'output_channels': [32, 32, 64, 64, 128, 128],
                 'kernel_shape': [[4, 4], [4, 4], [2, 3], [3, 3], [3, 3], [3, 3]],
                 'kernel_stride': [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
                 'pool_shape': [[2, 2], [2, 2], [2, 2], None, None, [2, 2]],
                 'pool_stride': [[2, 2], [2, 2], [2, 2], None, None, [2, 2]],
                 'weight_decay': [.0001, .0001, .0001, .0001, .0001, .0001, .0001],
                 'pool_type': ['max', 'max', 'max', None, None, 'max']}

   full_layers = {'nb_layers': 2,
                  'dim': [128, 128],
                  'weight_decay': [.0001, .0001],
                  'drop_out_probs': [.5, .5],
                  'drop_out_scales': [2., 2.]}

   optimization = ['rmsprop', 'adagrad', 'nesterov', 'momentum']

   for i in range(len(optimization)):

      best_result = optimization[i]+'_best_result.pkl'
      results = optimization[i]+'_result.pkl'


      # Build model
      model = CNN.build_model(train=train,
                              validation=valid,
                              test=test,
                              crop_size=crop_size,
                              conv_layers=convlayers,
                              fully_connected_layers=full_layers,
                              optimization=optimization[i],
                              use_weight_decay=True,
                              use_drop_out=False,
                              batch_size=100,
                              best_result_file=best_result,
                              results_file=results,
                              monitor_results=True,
                              momentum={'initial_value': 0.1, 'start': 1, 'saturate': 20, 'final_value': 0.99},
                              learning_rate={'initial_value': 0.1, 'start': 1, 'saturate': 20, 'decay_factor': 0.01})

      try:
         # Run model
         CNN.run(model)
      except (KeyboardInterrupt, SystemExit):
         raise
      except:
         print "Smells fishy! But continue anyways"



# run crop_whiten_and_save
# crop_whiten_and_save()









