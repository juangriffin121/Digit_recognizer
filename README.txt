Tips for training a model:
	-Its better to start using a small batch of the training data around 200 digits and training the model until it gets to around 80% to 90%, preferably not much more so as to not overfit the data and then go on to a larger traning batch and repeating the process until the percentage of correctly predicted digits is similar to the one of the previous iteration at which point theres probably no longer a problem with using any batch size.

ToDo:
	-make preprocess_data better
	-commit to github asap
	-make sesgos optional in both convolutional and dense layers
	-make shape of next layer input automatic
	-fix max pooling
	-test network with maxpool and 2 conv layers w/o sesgos