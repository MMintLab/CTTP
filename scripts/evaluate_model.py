


if __name__ == '__main__':
    import argparse
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='CMTJE')
    parser.add_argument('--dataset_name', type=str, default='dataset_2', help='Name of the dataset')
    parser.add_argument('--model_name', type=str, default='simclr', help='Name of the model')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')