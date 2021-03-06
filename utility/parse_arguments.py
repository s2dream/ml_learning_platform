import argparse

class ParseArgs:
    @classmethod
    def parse_arguments(cls):
        parser = argparse.ArgumentParser(description='Process some integers.')
        parser.add_argument("--model", default="", required=False)
        parser.add_argument("--mode", default="", required=False)
        parser.add_argument("-d", "--dist", action='store_true')
        return parser.parse_args()