from torch.nn import Transformer



x = Transformer.generate_square_subsequent_mask(10)

print(x)

