import ConvNet as CNN
import numpy as np

# Create dataset
crop_size = 200
train, valid = CNN.build_dogs_vs_cats_dataset(crop_size)

# Create model
# TODO: these tests should be ran automatically

# Build a dictionnary of different tests to run
kernel_shapes = np.arange(5,9,2)

output_channels_convnet = [[16, 16, 16], [32, 32, 32]]
output_channels_fullnet = [[16, 16, 16], [32, 32, 32], [64, 64, 64]]
pool_shapes = [[2, 2, 2]]
pool_strides = [[2, 2, 2]]


print kernel_shapes

conv_layers_tests = []
full_layers_tests = []
conv_test_idx = 0
max_pooling_shape = 2

# create conv layers tests
for i in xrange(len(kernel_shapes)):   # This loops on the dimension of the first layer convnet
   j = 0
   while j<i:  # We test until the dimension of the subsequent layers is as big as the one of the first layer

      kernel_shape = [[kernel_shapes[i], kernel_shapes[i]],
                      [kernel_shapes[j], kernel_shapes[j]],
                      [kernel_shapes[j], kernel_shapes[j]]]

      kernel_stride = []

      kernel_stride_idx = 1

      while kernel_stride_idx <= kernel_shapes[j]:
         kernel_stride = [(kernel_stride_idx, kernel_stride_idx),
                          (kernel_stride_idx, kernel_stride_idx),
                          (kernel_stride_idx, kernel_stride_idx)]

         pool_shape = []
         pool_max_value = min(kernel_shapes[j], max_pooling_shape)

         kernel_stride_idx += 1


         for out_channels_idx in range(len(output_channels_convnet)):

            conv_layers_tests.extend([{'nb_layers': 3,
                                      'output_channels': output_channels_convnet[out_channels_idx],
                                      'kernel_shape': kernel_shape,
                                      'kernel_stride': kernel_stride,
                                      'pool_shape': [[2, 2], [2, 2], [2, 2]],
                                      'pool_stride': [[2, 2], [2, 2], [2, 2]],
                                      'weight_decay': [.0001, .0001, .0001]}])

            conv_test_idx += 1

      j += 1


# create fully connected layers tests
for out_channels_idx in range(len(output_channels_fullnet)):
   full_layers_tests.extend([{'nb_layers': 3,
                              'dim': output_channels_fullnet[out_channels_idx],
                              'weight_decay': [.0001, .0001, .0001]}])



print "Total number of tests: ", len(conv_layers_tests)*len(full_layers_tests)

# run the tests with different configurations
test_idx = 0
for i in range(len(conv_layers_tests)):
   for j in range(len(full_layers_tests)):

      test_idx += 1
      test_name = 'test_{}_best_result'.format(test_idx)

      # Build model
      model = CNN.build_model(train=train,
                              validation=valid,
                              crop_size=crop_size,
                              conv_layers=conv_layers_tests[i],
                              fully_connected_layers=full_layers_tests[j],
                              use_weight_decay=True,
                              use_drop_out=False,
                              batch_size=50,
                              best_result_file=test_name,
                              monitor_results=False)

      # Run model
      CNN.run(model)