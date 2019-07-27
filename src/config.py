import os


DIR_DATA = './e-sun-ai-house-price'
TRAIN_DATASET = os.path.join(DIR_DATA, 'train.csv')
TEST_DATASET = os.path.join(DIR_DATA, 'test.csv')

NUM_CV = 5
N_BAGS = 8

AVAIL_GPU_LIST = [0, 3, 5, 6, 7]
