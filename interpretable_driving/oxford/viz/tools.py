


def generate_args():
    import argparse
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument(dest='partial_dir', default='None', help='dir.')

    args = argparser.parse_args()
    return args




