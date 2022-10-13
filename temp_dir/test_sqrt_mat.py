import torch


tensor_list = [[0, 1, 0, 1],
               [1, 0, 1, 0],
               [0, 1, 0, 1],
               [1, 0, 1, 0]]

t = torch.tensor(tensor_list)
t = t * 10
t = torch.sqrt(t)
print(t)
t = 1 / t
print(t)

def get_sqrt_and_reverse(X):
    X = torch.sqrt(X)
    X = 1/X
    return X

print("\n\n")

N=6
A = torch.randint(100,(N,N))
print(A)

I = torch.eye(N)
print(I)


A_tilde = A+I
print(A_tilde)

sum_row = torch.sum(A_tilde, -1)
print(sum_row)

D = torch.diag(sum_row)
print(D)

D_tilde = get_sqrt_and_reverse(D)