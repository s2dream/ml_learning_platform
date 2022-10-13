from torch.utils.data import BatchSampler
from torch.utils.data.distributed import DistributedSampler

a= DistributedSampler(range(80), num_replicas=8, rank=0, shuffle=False)
print(list(a))

a =list(BatchSampler(DistributedSampler(range(80), num_replicas=8, rank=0, shuffle=False), batch_size=5, drop_last=True))

print(a)
