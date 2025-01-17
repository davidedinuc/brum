import argparse

def get_params():
    parser = argparse.ArgumentParser()

    # general params
    parser.add_argument('--seed', type=int, default=1518,
                        help='seed')
    
    
    return parser.parse_args()