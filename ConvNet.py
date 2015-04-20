# This class implements the ConvNet class on whose model our classification
# will be based
from pylearn2.models.mlp import MLP
from pylearn2.models.mlp import Softmax
from pylearn2.models.mlp import RectifiedLinear
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
import pylearn2.training_algorithms.learning_rule as learning_rule
import pylearn2.training_algorithms.sgd as sgd
# from Preprocessing import Preprocess

def build_dogs_vs_cats_dataset(crop_size=200):
   scaled_size = crop_size + 25

   # Crop the image randomly in a [crop_size, crop_size] size
   rand_crop = RandomCrop(scaled_size=scaled_size, crop_size=crop_size)

   # Train set
   train = DogsVsCats(transformer=rand_crop, start=0, stop=20000)

   # Validation set
   validation = DogsVsCats(transformer=rand_crop, start=20000, stop=22500)

   # Test set
   test = DogsVsCats(transformer=rand_crop, start=22500, stop=25000)

   return train, validation, test


def build_model(train,
                validation,
                test,
                crop_size,
                conv_layers,
                fully_connected_layers,
                use_weight_decay,
                use_drop_out,
                momentum,
                learning_rate,
                batch_size=100,
                max_epochs=1000,
                monitor_results=True,
                optimization='momentum',
                results_file='convnet_results.pkl',
                best_result_file='convnet_best_result.pkl'):

   # TODO: add some parameters validation

   # Construct model

   # Convolution layers
   weight_decay_coeffs = {}
   nb_conv_layers = conv_layers['nb_layers']
   hidden_conv_layers = [0]*nb_conv_layers
   for i in range(nb_conv_layers):
      layer_name = 'h_c_{}'.format(i+1)

      print conv_layers['kernel_stride'][i]

      hidden_conv_layers[i] = ConvRectifiedLinear(layer_name=layer_name,
                                                  output_channels=conv_layers['output_channels'][i],
                                                  irange=0.1,
                                                  kernel_shape=conv_layers['kernel_shape'][i],
                                                  pool_shape=conv_layers['pool_shape'][i],
                                                  pool_stride=conv_layers['pool_stride'][i],
                                                  pool_type=conv_layers['pool_type'][i],
                                                  kernel_stride=conv_layers['kernel_stride'][i])
      
      if(use_weight_decay):
          weight_decay_coeffs[layer_name] = conv_layers['weight_decay'][i]

   layers = []

   layers.extend(hidden_conv_layers)


   # Fully connected layers
   nb_fully_connected_layers = fully_connected_layers['nb_layers']
   hidden_full_layers = [0] * nb_fully_connected_layers
   drop_out_probs = {}
   drop_out_scales = {}
   for i in range(nb_fully_connected_layers):

      layer_name = 'h_f_{}'.format(i+1)
      hidden_full_layers[i] = RectifiedLinear(layer_name=layer_name,
                                              irange=0.1,
                                              dim=fully_connected_layers['dim'][i])

      if (use_weight_decay):
         weight_decay_coeffs[layer_name] = fully_connected_layers['weight_decay'][i]

      if(use_drop_out):
         drop_out_probs[layer_name] = fully_connected_layers['drop_out_probs'][i]
         drop_out_scales[layer_name] = fully_connected_layers['drop_out_scales'][i]


   layers.extend(hidden_full_layers)

   # Build output layer
   output_layer = Softmax(layer_name='output',
                          n_classes=2,
                          irange=0.1)

   layers.extend([output_layer])


   model = MLP(batch_size=batch_size,
               input_space=Conv2DSpace(shape=[crop_size, crop_size], num_channels=3),
               layers=layers)

   # Construct training (or optimization?) algorithm object
   cost_methods_wanted = [MethodCost(method='cost_from_X')]

   if use_weight_decay:
      # add weight decay for output layer
      weight_decay_coeffs['output']=.00001 # TODO: this should also be a paramter
      cost_methods_wanted.extend([WeightDecay(coeffs=weight_decay_coeffs)])

   if use_drop_out:
      cost_methods_wanted.extend([Dropout(default_input_include_prob=1.0,
                                          default_input_scale=1.0,
                                          input_include_probs=drop_out_probs,
                                          input_scales=drop_out_scales)])

   cost_methods = SumOfCosts(costs=cost_methods_wanted)

   termination_criteria = And([MonitorBased(channel_name='valid_output_misclass',
                                            prop_decrease=0.01,
                                            N=20),             # number of epochs to look back
                               EpochCounter(max_epochs=max_epochs)])

   # optimization
   extensions = [MonitorBasedSaveBest(channel_name='valid_output_misclass',
                                      save_path=best_result_file),
                 sgd.LinearDecayOverEpoch(start=learning_rate['start'],
                                          saturate=learning_rate['saturate'],
                                          decay_factor=learning_rate['decay_factor'])]

   optimization_rule = []

   if optimization == 'nesterov':
      optimization_rule = learning_rule.Momentum(init_momentum=momentum['initial_value'],
                                                 nesterov_momentum=True)
      extensions.extend([learning_rule.MomentumAdjustor(start=momentum['start'],
                                                        saturate=momentum['saturate'],
                                                        final_momentum=momentum['final_value'])])
   elif optimization == 'momentum':
      optimization_rule = learning_rule.Momentum(init_momentum=momentum['initial_value'])
      extensions.extend([learning_rule.MomentumAdjustor(start=momentum['start'],
                                                        saturate=momentum['saturate'],
                                                        final_momentum=momentum['final_value'])])
   elif optimization == 'rmsprop':
      optimization_rule = learning_rule.RMSProp()

   elif optimization == 'adagrad':
      optimization_rule = learning_rule.AdaGrad()


   algorithm = sgd.SGD(batch_size=batch_size,
                       train_iteration_mode='batchwise_shuffled_sequential',
                       batches_per_iter=100,
                       monitoring_batch_size=batch_size,
                       monitoring_batches=10,
                       monitor_iteration_mode='batchwise_shuffled_sequential',
                       learning_rate=learning_rate['initial_value'],
                       learning_rule=optimization_rule,
                       monitoring_dataset={'train': train,
                                           'valid': validation,
                                           'test': test},
                       cost=cost_methods,
                       termination_criterion=termination_criteria)

   # Run test
   if monitor_results:
      cnn = Train(dataset=train,
                  model=model,
                  algorithm=algorithm,
                  save_path=results_file,
                  save_freq=20,
                  extensions=extensions)
   else:
      cnn = Train(dataset=train,
                  model=model,
                  algorithm=algorithm,
                  save_freq=0,
                  extensions=extensions)

   return cnn

def run(model):
   model.main_loop()
