import random 
from os import listdir
from os.path import isfile, join

PATH = './data/cat_vs_dog/'
TRAIN_PARTITION = 0.7 
VALIDATION_PARTITION = 0.15 
TEST_PARTITION = 0.15 

def data_to_textfile(data, type):
    # Open file. 
    filename = type + '.txt'
    file = open(filename, 'w')

    # Write to file. 
    for example in data: 
        example_split = example.split('.')

        if example_split[0] == 'cat': 
            line = example + ' 0'
        else: 
            line = example + ' 1'
        
        file.write(PATH + '%s\n' % line)

    # Close file. 
    file.close()

def partition(data): 
    total_instances = len(data)
    train_end_idx = int(TRAIN_PARTITION * total_instances)
    val_end_idx = train_end_idx + int(VALIDATION_PARTITION * total_instances) 
    test_end_idx = val_end_idx + int(TEST_PARTITION * total_instances) 

    train_data = data[ : train_end_idx]
    validation_data = data[train_end_idx : val_end_idx]
    test_data = data[val_end_idx : ]

    return train_data, validation_data, test_data

def main(): 
    # Store string of image names 
    image_list = [f for f in listdir(PATH) if isfile(join(PATH, f))]

    # Shuffle the data 
    random.shuffle(image_list)

    # Partition the data 
    train_data, validation_data, test_data = partition(image_list)

    # Data to text file
    data_to_textfile(train_data, type='train')
    data_to_textfile(validation_data, type='validation')
    data_to_textfile(test_data, type='test')

if __name__=='__main__': 
    main() 