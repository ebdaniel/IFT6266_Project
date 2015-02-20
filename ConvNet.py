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
from pylearn2.termination_criteria import EpochCounter
from pylearn2.termination_criteria import MonitorBased
from pylearn2.termination_criteria import And
from pylearn2.train import Train
from pylearn2.models.mlp import Conv2DSpace
from pylearn2.models.mlp import ConvRectifiedLinear
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest


from ift6266h15.code.pylearn2.datasets.variable_image_dataset import DogsVsCats, RandomCrop

# Load dataset
crop_size = 221
scaled_size = 256
rand_crop = RandomCrop(scaled_size=scaled_size, crop_size=crop_size)

train = DogsVsCats(transformer=rand_crop,
                   start=0,
                   stop=20000)

validation = DogsVsCats(transformer=rand_crop,
                        start=20000,
                        stop=22500)

# Construct model
batch_size = 100
hidden_layer_1 = ConvRectifiedLinear(layer_name='h_1',
                                     output_channels=64,
                                     irange=.05,
                                     kernel_shape=[5,5],
                                     pool_shape=[4,4],
                                     pool_stride=[2,2],
                                     max_kernel_norm=1.9365)

hidden_layer_2 = ConvRectifiedLinear(layer_name='h_2',
                                     output_channels=64,
                                     irange=.05,
                                     kernel_shape=[5,5],
                                     pool_shape=[4,4],
                                     pool_stride=[2,2],
                                     max_kernel_norm=1.9365)

hidden_layer_3 = ConvRectifiedLinear(layer_name='h_3',
                                     output_channels=64,
                                     irange=.05,
                                     kernel_shape=[5, 5],
                                     pool_shape=[4, 4],
                                     pool_stride=[2, 2],
                                     max_kernel_norm=1.9365)

hidden_layer_4 = RectifiedLinear()
hidden_layer_5 = RectifiedLinear()
hidden_layer_6 = RectifiedLinear()

output_layer = Softmax(max_col_norm=1.9365,
                       layer_name='output',
                       n_classes=2,
                       istdev=.05)

model = MLP(batch_size=batch_size,
            input_space=Conv2DSpace(shape=[crop_size,crop_size], num_channels=3),
            layers=[hidden_layer_1, hidden_layer_2, hidden_layer_3,
                    hidden_layer_4, hidden_layer_5, hidden_layer_6, output_layer])

# Construct training (or optimization?) algorithm object
cost_method_1 = MethodCost(method='cost_from_X')
cost_method_2 = WeightDecay(coeffs=[.00005, .00005, .00005])

cost_methods = SumOfCosts(costs=[cost_method_1, cost_method_2])

termination_criteria = And([MonitorBased(channel_name='valid_output_misclass',
                                         prop_decrease=0.50,
                                         N=10),
                            EpochCounter(max_epochs=10)])

algorithm = SGD(batch_size=batch_size,
                train_iteration_mode='batchwise_shuffled_sequential',
                batches_per_iter=10,
                monitoring_batch_size=batch_size,
                monitoring_batches=10,
                monitor_iteration_mode='batchwise_shuffled_sequential',
                learning_rate=1e-3,
                learning_rule=Momentum(init_momentum=0.95),
                monitoring_dataset={'train':train,
                                    'valid':validation},
                cost=cost_methods,
                termination_criterion=termination_criteria)

extensions = [MonitorBasedSaveBest(channel_name='valid_output_misclass',
                                   save_path="convnetBestResults.pkl"),
              MomentumAdjustor(start=1,
                               saturate=10,
                               final_momentum=.99)]


# Run test
train = Train(dataset=train,
              model=model,
              algorithm=algorithm,
              save_path='convnetResults.pkl',
              save_freq=1,
              extensions=extensions)

train.main_loop()
