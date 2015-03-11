import ConvNet as CNN

# Create dataset
crop_size = 224
train, valid, test = CNN.build_dogs_vs_cats_dataset(crop_size=crop_size)

# Create model
convlayers = {'nb_layers': 5,
              'output_channels': [32, 32, 128, 128, 32],
              'kernel_shape': [[5, 5], [5, 5], [4, 4], [4, 4], [4, 4]],
              'kernel_stride': [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
              'pool_shape': [[2, 2], [2, 2], None, None, [2, 2]],
              'pool_stride': [[2, 2], [2, 2], None, None, [2, 2]],
              'weight_decay': [.0001, .0001, .0001, .0001, .0001, .0001],
              'pool_type': ['max', 'max', None, None, 'max']}

full_layers = {'nb_layers': 2,
               'dim': [128, 128],
               'weight_decay': [.0001, .0001],
               'drop_out_probs': [.5, .5],
               'drop_out_scales': [2., 2.]}

model = CNN.build_model(train=train,
                        validation=valid,
			               test=test,
                        crop_size=crop_size,
                        conv_layers=convlayers,
                        fully_connected_layers=full_layers,
                        batch_size=50,
                        use_weight_decay=True,
                        use_drop_out=False)

# Run the model
CNN.run(model)
