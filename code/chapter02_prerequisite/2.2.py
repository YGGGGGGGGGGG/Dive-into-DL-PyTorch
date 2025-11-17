import torch
import numpy as np

def operation_data():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    print(torch.__version__)

def create_tensor(): # 2.2.1创建张量
    x = torch.empty(5, 3, dtype=torch.long) # 创建5x3的全0张量
    y = torch.rand(3, 5) # 创建5x3的随机张量
    z = torch.tensor([[5.3, 6.0, 0], [1, 2, 3], [2.0, 0, 0]])
    print(x, x.shape)
    x = x.new_ones(5, 3, dtype=torch.float64)
    print(x, x.shape)
    x = torch.randn_like(x, dtype=torch.float)
    print(x, x.shape)
    print(y, y.shape)
    print(z, z.shape)

def operation_tensor():
    x = torch.ones(5, 3)
    y = torch.ones(5, 3)
    z = torch.ones(3, 5)
    c = torch.add(x, y) # 创建一个新的张量
    d = y.add_(x) # 原地操作
    print(torch.matmul(x, z))
    result = torch.empty(5, 3)
    new_result = torch.add(x, y, out=result) # out 是用了旧内存，优化性能
    print(new_result)
    y = x[0, :]
    y += 1
    print(y)
    print(x[0, :]) # 张量里面的切片操作，逗号控制的是维度，逗号最左边的行，第二个是列
    # print(x.item()) # 只能用于单张量


def broadcast_(): # 2.2.3广播机制
    x = torch.arange(1, 3).view(1, 2)
    print(x)
    y = torch.arange(1, 4) # 生成一维的1到3
    print(y)
    y = y.view(3, 1)
    print(y)
    print(x + y)

def calculate_size(): # 2.2.4
    x = torch.tensor([1, 2])
    y = torch.tensor([3, 4])
    z = torch.tensor([3, 4])
    id_before_z = id(z) # id相当于对象的身份证号码，唯一
    id_before = id(y)
    print(id(y) == id_before) # true
    y = x + y
    print(id(y) == id_before) # false
    z[:] = z + x # 创建一个临时对象，但使用原内存
    print(id(z) == id_before_z) # true

def tensor_to_numpy(): # 2.2.5
    a = torch.ones(5)
    b = a.numpy()
    print(a, b)
    a += 1
    print(a, b)
    b += 1
    print(a, b)


def numpy_to_tensor():
    a = np.ones(5)
    b = torch.from_numpy(a)
    print(a, b)
    a += 1
    print(a, b)
    b += 1
    print(a, b)
    c = torch.tensor(a) # 这个和上面的区别在于，这个是拷贝，不会共享内存
    a += 1
    print(b, c)
    print("NumPy数组的数据指针:", a.ctypes.data)
    print("PyTorch张量的数据指针:", b.data_ptr())
    print("是否共享底层内存:", a.ctypes.data == b.data_ptr())

def GPU_is_available():
    print("CUDA 可用:", torch.cuda.is_available())
GPU_is_available()
def tensor_on_gpu(): # 2.2.6 在GPU运行
    if torch.cuda.is_available():
        device = torch.device("cuda") # GPU
        x = torch.tensor([1, 2])
        y = torch.ones_like(x, device=device)
        x = x.to(device)
        z = x + y
        print(z)
        print(z.to("cpu", torch.double))
tensor_on_gpu()

