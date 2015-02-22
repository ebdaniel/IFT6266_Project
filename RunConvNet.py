import ConvNet as CNN

# Create dataset
crop_size = 200
train, valid = CNN.build_dogs_vs_cats_dataset(crop_size)

# Create model
# TODO: these tests should be ran automatically
convlayers = {'nb_layers':3,
              'output_channels':[32, 16, 16],
              'kernel_shape':[[11, 11], [7, 7], [5, 5]],
              'pool_shape':[[4,4],[4,4],[4,4]],
              'pool_stride':[[2,2],[2,2],[2,2]],
              'weight_decay':[.00005, .00005, .00005]}

full_connected_layers ={'nb_layers':3,
                        'dim':[64,64,64],
                        'weight_decay':[.00005, .00005, .00005]}

model = CNN.build_model(train=train,
                        validation=valid,
                        crop_size=crop_size,
                        conv_layers=convlayers,
                        fully_connected_layers=full_connected_layers,
                        use_weight_decay=True,
                        use_drop_out=True)

# Run model
CNN.run(model)