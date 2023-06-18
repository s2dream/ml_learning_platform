import time
import torch
from model.transformer.transformer_overall import Transformer
from log_module.ml_logger import MLLogger

DEVICE = "cpu"

def test_transformer():
    logger = MLLogger.get_logger()
    transformer_model = Transformer(nhead=16, num_encoder_layers=6).to(DEVICE)
    src = torch.rand((32, 512, 512)).to(DEVICE)
    tgt = torch.rand((32, 512, 512)).to(DEVICE)
    stime = time.time()
    out = transformer_model(src, tgt).cpu()
    time.sleep(5)
    logger.info(out)
    logger.info(time.time()-stime)


if __name__ == "__main__":
    test_transformer()
