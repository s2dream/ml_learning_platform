import argparse


class ParseArgs:
    @classmethod
    def parse_arguments(cls):
        parser = argparse.ArgumentParser(description='Arguments: —task [task name] —model [model name] --optimizer [optimizer name] --dataloader [dataloader name] [option:-d]')
        parser.add_argument("--task", default="", required=True)
        parser.add_argument("--model", default="", required=True)
        parser.add_argument("--optimizer", default="AdamW", required=False)
        parser.add_argument("--dataloader", default="", required=True)
        parser.add_argument("-d", "--dist", action='store_true')
        return parser.parse_args()

