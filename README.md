# Transfer Learning within MNIST 

In this module, we conduct transfer learning within MNIST. We split the entire dataset into two groups: 1-4 and 5-9. Then we transfer information learned from the first group (1-4) and retrain the network on the second dataset (5-9) to see if there are any improvements on the new and previous datasets. 

## Training the first dataset 

To train the first dataset (1-4), open a fresh terminal and run `python train.py` (You should be in the `mnist` directory before executing the command. The model will be saved in `/convnet_mnist`. 

### Launching TensorBoard  

If you would like to visualize results in TensorBoard, open a separate terminate, change directory to inside `mnist`, and run the following command: 

`tensorboard --logdir=./graphs/convnet`

## Training the second dataset (transfer baseline)

First comment out line 21 and uncomment line 22 in `train.py`. Follow the same procedure for training and visualizing as outlined in the previous section. 

## Transferring from original dataset to new dataset 

Train the network as outlined in the first section ('Training the first dataset'). Make sure line 21 is uncommented and line 22 is commented out. Once finished, go to line 23 of `train.py` and set `retrain` to `True`. Run `python train.py` in the `mnist` directory. 

Again if you would like to visualize results, follow the procedure as outlined above. The final results should display in the terminal.

## Testing on the original dataset after transfer. 

Follow the procedure above. Once done, set `retrain` in line 23 of `train.py` to `False` and train the model on the original dataset (see directions in 'Training the first dataset'). The new performance on the original task should decrease. 


