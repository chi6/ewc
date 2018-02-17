#Some changes in this code

1. Integrated the config file in all the .py files, the main skeleton is unchanged
    To set config:
        1. You can set the default parameters in 'config.py' file
        2. Also, you can run 'config.py -h' to get all parameters options
        3. To set the current parameter, run like 'config.py --dataset mnist'


2. Set up two new files, transfer_l2.py and post_transfer_l2.py, which contain L2 loss in transfer process:
    To run these L2 loss files:
        To train the first dataset (0-4), open a fresh terminal and run `python baseline_0-4.py`
        To transfer to the second dataset (5-9), run 'transfer_l2.py'
        To test in first dataset (0-4), run 'post_transfer_l2.py'
    config options:
        -lambda_l2  ,default is 0.01

    Results:
        As we expected, when I set the lambda_l2 as 1e7, the test accuracy in original dataset(0-4) remains as 0.227,
    which is about the same as the previous accuracy 0.226

3. There is still some bugs in the 'ewc' files. I also integrated the config in it. If you have some more options,
    please tell me and I will do more adjustments.
