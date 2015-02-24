# This class implements the ConvNet class on whose model our classification
# will be based
from pylearn2.models.mlp import MLP
from pylearn2.models.mlp import Softmax
from pylearn2.models.mlp import RectifiedLinear
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.training_algorithms.learning_rule import Momentum
from pylearn2.training_algorithms.learning_rule import MomentumAdjustor
from pylearn2.costs.cost import MethodCost
from pylearn2.costs.cost import SumOfCosts
from pylearn2.costs.mlp import WeightDecay
from pylearn2.costs.mlp.dropout import Dropout
from pylearn2.termination_criteria import EpochCounter
from pylearn2.termination_criteria import MonitorBased
from pylearn2.termination_criteria import And
from pylearn2.train import Train
from pylearn2.models.mlp import Conv2DSpace
from pylearn2.models.mlp import ConvRectifiedLinear
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest


from ift6266h15.code.pylearn2.datasets.variable_image_dataset import DogsVsCats, RandomCrop

def build_dogs_vs_cats_dataset(crop_size=200):
   scaled_size = crop_size+20

   # Crop the image randomly in a [crop_size, crop_size] size
   rand_crop = RandomCrop(scaled_size=scaled_size, crop_size=crop_size)

   # Train set
   train = DogsVsCats(transformer=rand_crop, start=0, stop=20000)

   # Validation set
   validation = DogsVsCats(transformer=rand_crop, start=20000, stop=22500)

   return [ train, validation]


def build_model(train,
                validation,
                crop_size,
                conv_layers,
                fully_connected_layers,
                use_weight_decay,
                use_drop_out,
                best_result_file,
                batch_size=100,
                max_epochs=1000,
                monitor_results = True,
                results_file = 'convnet_results.pkl'):

   # TODO: add some parameters validation

   # Construct model

   # Convolution layers
   weight_decay_coeffs = {}
   nb_conv_layers = conv_layers['nb_layers']
   hidden_conv_layers = [0]*nb_conv_layers
   for i in range(nb_conv_layers):
      layer_name = 'h_c_{}'.format(i+1)
      hidden_conv_layers[i] = ConvRectifiedLinear(layer_name=layer_name,
                                                  output_channels=conv_layers['output_channels'][i],
                                                  irange=.01,
                                                  kernel_shape=conv_layers['kernel_shape'][i],
                                                  pool_shape=conv_layers['pool_shape'][i],
                                                  pool_stride=conv_layers['pool_stride'][i])

      weight_decay_coeffs = {layer_name:conv_layers['weight_decay'][i]}

   layers = []

   layers.extend(hidden_conv_layers)


   # Fully connected layers
   nb_fully_connected_layers = fully_connected_layers['nb_layers']
   hidden_full_layers = [0] * nb_fully_connected_layers
   for i in range(nb_fully_connected_layers):

      layer_name = 'h_f_{}'.format(i+1)
      hidden_full_layers[i] = RectifiedLinear(layer_name=layer_name,
                                              irange=.01,
                                              dim=fully_connected_layers['dim'][i])

      if (use_weight_decay):
         weight_decay_coeffs = {layer_name: fully_connected_layers['weight_decay'][i]}


   layers.extend(hidden_full_layers)

   # Build output layer
   output_layer = Softmax(max_col_norm=1.9365,
                          layer_name='output',
                          n_classes=2,
                          istdev=.05)

   layers.extend([output_layer])


   print len(layers)

   model = MLP(batch_size=batch_size,
               input_space=Conv2DSpace(shape=[crop_size,crop_size], num_channels=3),
               layers=layers)

   # Construct training (or optimization?) algorithm object
   cost_methods_wanted = [MethodCost(method='cost_from_X')]

   if use_weight_decay:
      # add weight decay for output layer
      weight_decay_coeffs['output']=.00005 # TODO: this should also be a paramter
      cost_methods_wanted.extend([WeightDecay(coeffs=weight_decay_coeffs)])

   if use_drop_out:
      cost_methods_wanted.extend([Dropout()])

   cost_methods = SumOfCosts(costs=cost_methods_wanted)

   termination_criteria = And([MonitorBased(channel_name='valid_output_misclass',
                                            prop_decrease=0.10,
                                            N=10),             # number of epochs to look back
                               EpochCounter(max_epochs=max_epochs)])

   algorithm = SGD(batch_size=batch_size,
                   train_iteration_mode='even_batchwise_shuffled_sequential',
                   batches_per_iter=10,
                   monitoring_batch_size=batch_size,
                   monitoring_batches=batch_size,
                   monitor_iteration_mode='even_batchwise_shuffled_sequential',
                   learning_rate=1e-3,
                   learning_rule=Momentum(init_momentum=0.1),
                   monitoring_dataset={'train':train,
                                       'valid':validation},
                   cost=cost_methods,
                   termination_criterion=termination_criteria)

   extensions = [MonitorBasedSaveBest(channel_name='valid_output_misclass',
                                      save_path=best_result_file),
                 MomentumAdjustor(start=10,
                                  saturate=100,
                                  final_momentum=.99)]


   # Run test

   if monitor_results:
      train = Train(dataset=train,
                    model=model,
                    algorithm=algorithm,
                    save_path=results_file,
                    save_freq=10,
                    extensions=extensions)
   else:
      train = Train(dataset=train,
                    model=model,
                    algorithm=algorithm,
                    save_freq=0,
                    extensions=extensions)

   return train

def run(model):
   model.main_loop()
