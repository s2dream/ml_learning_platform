import time
import torch
from model.transformer.transformer_overall import Transformer


def test_transformer():
    transformer_model = Transformer(nhead=16, num_encoder_layers=6).to("cuda")
    src = torch.rand((32, 512, 512)).to("cuda")
    tgt = torch.rand((32, 512, 512)).to("cuda")
    stime = time.time()
    out = transformer_model(src, tgt).cpu()
    time.sleep(5)
    print(out)
    print(time.time()-stime)


if __name__ == "__main__":
    test_transformer()
