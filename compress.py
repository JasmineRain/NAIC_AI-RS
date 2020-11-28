import numpy as np
from sklearn.cluster import KMeans

import os
from collections import defaultdict, namedtuple
from heapq import heappush, heappop, heapify
import struct
from pathlib import Path

import torch
import torch.nn as nn

from scipy.sparse import csr_matrix, csc_matrix

from models.DeepLabV3_plus.deeplabv3_plus import DeepLabv3_plus
from models import DANet


def apply_weight_sharing(model, bits):
    """
    Applies weight sharing to the given model
    """
    # for module in model.children():
    for name, module in model.named_modules():
        # print(name)
        if isinstance(module, torch.nn.Conv2d):
            # bits = 5
            bits = bits
        elif isinstance(module, torch.nn.Linear):
            # bits = 5
            bits = bits
        else:
            continue
        dev = module.weight.device
        weight = module.weight.data.cpu().numpy()
        ori_shape = weight.shape
        if len(weight.shape) != 2:
            length = len(weight.flatten())
            num = np.arange(int(np.sqrt(length)), length + 1)
            mask = length % num == 0
            num = num[mask]
            weight = weight.reshape(-1, num[0])
        shape = weight.shape
        mat = csr_matrix(weight) if shape[0] < shape[1] else csc_matrix(weight)
        min_ = min(mat.data)
        max_ = max(mat.data)
        space = np.linspace(min_, max_, num=2 ** bits)
        kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1, 1), n_init=1,
                        algorithm="full")
        kmeans.fit(mat.data.reshape(-1, 1))
        new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
        mat.data = new_weight
        module.weight.data = torch.from_numpy(mat.toarray()).to(dev).reshape(ori_shape)


Node = namedtuple('Node', 'freq value left right')
Node.__lt__ = lambda x, y: x.freq < y.freq


def huffman_encode(arr, prefix, save_dir):
    """
    Encodes numpy array 'arr' and saves to `save_dir`
    The names of binary files are prefixed with `prefix`
    returns the number of bytes for the tree and the data after the compression
    """
    # Infer dtype
    dtype = str(arr.dtype)

    # Calculate frequency in arr
    freq_map = defaultdict(int)
    convert_map = {'float32': float, 'int32': int}
    for value in np.nditer(arr):
        value = convert_map[dtype](value)
        freq_map[value] += 1

    # Make heap
    heap = [Node(frequency, value, None, None) for value, frequency in freq_map.items()]
    heapify(heap)

    # Merge nodes
    while (len(heap) > 1):
        node1 = heappop(heap)
        node2 = heappop(heap)
        merged = Node(node1.freq + node2.freq, None, node1, node2)
        heappush(heap, merged)

    # Generate code value mapping
    value2code = {}

    def generate_code(node, code):
        if node is None:
            return
        if node.value is not None:
            value2code[node.value] = code
            return
        generate_code(node.left, code + '0')
        generate_code(node.right, code + '1')

    root = heappop(heap)
    if root.left is None and root.right is None:
        value2code[root.value] = '0'
    else:
        generate_code(root, '')

    # Path to save location
    directory = Path(save_dir)

    # Dump data
    data_encoding = ''.join(value2code[convert_map[dtype](value)] for value in np.nditer(arr))
    datasize = dump(data_encoding, directory / f'{prefix}.bin')

    # Dump codebook (huffman tree)
    codebook_encoding = encode_huffman_tree(root, dtype)
    treesize = dump(codebook_encoding, directory / f'{prefix}_codebook.bin')

    return treesize, datasize


def huffman_decode(directory, prefix, dtype):
    """
    Decodes binary files from directory
    """
    directory = Path(directory)

    # Read the codebook
    codebook_encoding = load(directory / f'{prefix}_codebook.bin')
    root = decode_huffman_tree(codebook_encoding, dtype)

    # Read the data
    data_encoding = load(directory / f'{prefix}.bin')

    # Decode
    data = []
    ptr = root
    for bit in data_encoding:
        if ptr.left is not None and bit == '0':
            ptr = ptr.left
        elif ptr.right is not None:
            ptr = ptr.right

        if ptr.value is not None:  # Leaf node
            data.append(ptr.value)
            ptr = root

    return np.array(data, dtype=dtype)


# Logics to encode / decode huffman tree
# Referenced the idea from https://stackoverflow.com/questions/759707/efficient-way-of-storing-huffman-tree
def encode_huffman_tree(root, dtype):
    """
    Encodes a huffman tree to string of '0's and '1's
    """
    converter = {'float32': float2bitstr, 'int32': int2bitstr}
    code_list = []

    def encode_node(node):
        if node.value is not None:  # node is leaf node
            code_list.append('1')
            lst = list(converter[dtype](node.value))
            code_list.extend(lst)
        else:
            code_list.append('0')
            encode_node(node.left)
            encode_node(node.right)

    encode_node(root)
    return ''.join(code_list)


def decode_huffman_tree(code_str, dtype):
    """
    Decodes a string of '0's and '1's and costructs a huffman tree
    """
    converter = {'float32': bitstr2float, 'int32': bitstr2int}
    idx = 0

    def decode_node():
        nonlocal idx
        info = code_str[idx]
        idx += 1
        if info == '1':  # Leaf node
            value = converter[dtype](code_str[idx:idx + 32])
            idx += 32
            return Node(0, value, None, None)
        else:
            left = decode_node()
            right = decode_node()
            return Node(0, None, left, right)

    return decode_node()


# My own dump / load logics
def dump(code_str, filename):
    """
    code_str : string of either '0' and '1' characters
    this function dumps to a file
    returns how many bytes are written
    """
    # Make header (1 byte) and add padding to the end
    # Files need to be byte aligned.
    # Therefore we add 1 byte as a header which indicates how many bits are padded to the end
    # This introduces minimum of 8 bits, maximum of 15 bits overhead
    num_of_padding = -len(code_str) % 8
    header = f"{num_of_padding:08b}"
    code_str = header + code_str + '0' * num_of_padding

    # Convert string to integers and to real bytes
    byte_arr = bytearray(int(code_str[i:i + 8], 2) for i in range(0, len(code_str), 8))

    # Dump to a file
    with open(filename, 'wb') as f:
        f.write(byte_arr)
    return len(byte_arr)


def load(filename):
    """
    This function reads a file and makes a string of '0's and '1's
    """
    with open(filename, 'rb') as f:
        header = f.read(1)
        rest = f.read()  # bytes
        code_str = ''.join(f'{byte:08b}' for byte in rest)
        offset = ord(header)
        if offset != 0:
            code_str = code_str[:-offset]  # string of '0's and '1's
    return code_str


# Helper functions for converting between bit string and (float or int)
def float2bitstr(f):
    four_bytes = struct.pack('>f', f)  # bytes
    return ''.join(f'{byte:08b}' for byte in four_bytes)  # string of '0's and '1's


def bitstr2float(bitstr):
    byte_arr = bytearray(int(bitstr[i:i + 8], 2) for i in range(0, len(bitstr), 8))
    return struct.unpack('>f', byte_arr)[0]


def int2bitstr(integer):
    four_bytes = struct.pack('>I', integer)  # bytes
    return ''.join(f'{byte:08b}' for byte in four_bytes)  # string of '0's and '1's


def bitstr2int(bitstr):
    byte_arr = bytearray(int(bitstr[i:i + 8], 2) for i in range(0, len(bitstr), 8))
    return struct.unpack('>I', byte_arr)[0]


# Functions for calculating / reconstructing index diff
def calc_index_diff(indptr):
    return indptr[1:] - indptr[:-1]


def reconstruct_indptr(diff):
    return np.concatenate([[0], np.cumsum(diff)])


# Encode / Decode models
def huffman_encode_model(model, directory='encodings/'):
    os.makedirs(directory, exist_ok=True)
    original_total = 0
    compressed_total = 0
    # print(f"{'Layer':<15} | {'original':>10} {'compressed':>10} {'improvement':>11} {'percent':>7}")
    # print('-' * 70)
    # for name, param in model.named_parameters():
    for name, param in model.state_dict().items():
        if 'weight' in name:
            weight = param.data.cpu().numpy()
            if len(weight.shape) != 2:
                length = len(weight.flatten())
                # weight = weight.reshape(-1, int(np.sqrt(len(weight.flatten()))))
                num = np.arange(int(np.sqrt(length)), length + 1)
                mask = length % num == 0
                num = num[mask]
                weight = weight.reshape(-1, num[0])
            # for weight in all_weight
            shape = weight.shape
            form = 'csr' if shape[0] < shape[1] else 'csc'
            mat = csr_matrix(weight) if shape[0] < shape[1] else csc_matrix(weight)

            # Encode
            t0, d0 = huffman_encode(mat.data, name + f'_{form}_data', directory)
            t1, d1 = huffman_encode(mat.indices, name + f'_{form}_indices', directory)
            t2, d2 = huffman_encode(calc_index_diff(mat.indptr), name + f'_{form}_indptr', directory)

            # Print statistics
            original = mat.data.nbytes + mat.indices.nbytes + mat.indptr.nbytes
            compressed = t0 + t1 + t2 + d0 + d1 + d2

            # print(
            #     f"{name:<15} | {original:10} {compressed:10} {original / compressed:>10.2f}x {100 * compressed / original:>6.2f}%")
        else:  # bias
            # Note that we do not huffman encode bias
            bias = param.data.cpu().numpy()
            bias.dump(f'{directory}/{name}')

            # Print statistics
            original = bias.nbytes
            compressed = original

            # print(
            #     f"{name:<15} | {original:10} {compressed:10} {original / compressed:>10.2f}x {100 * compressed / original:>6.2f}%")
        original_total += original
        compressed_total += compressed

    # print('-' * 70)
    # print(
    #     f"{'total':15} | {original_total:>10} {compressed_total:>10} {original_total / compressed_total:>10.2f}x {100 * compressed_total / original_total:>6.2f}%")


def huffman_decode_model(model, directory):
    # for name, param in model.named_parameters():
    new_state = model.state_dict()
    for name, param in new_state.items():
        print(name)
        if 'weight' in name:
            dev = param.device
            weight = param.data.cpu().numpy()
            tar_shape = weight.shape
            if len(weight.shape) != 2:
                length = len(weight.flatten())
                num = np.arange(int(np.sqrt(length)), length + 1)
                mask = length % num == 0
                num = num[mask]
                weight = weight.reshape(-1, num[0])
            shape = weight.shape
            form = 'csr' if shape[0] < shape[1] else 'csc'
            matrix = csr_matrix if shape[0] < shape[1] else csc_matrix

            # Decode data
            data = huffman_decode(directory, name + f'_{form}_data', dtype='float32')
            indices = huffman_decode(directory, name + f'_{form}_indices', dtype='int32')
            indptr = reconstruct_indptr(huffman_decode(directory, name + f'_{form}_indptr', dtype='int32'))

            # Construct matrix
            mat = matrix((data, indices, indptr), shape)

            # Insert to model
            new_weight = torch.from_numpy(mat.toarray()).to(dev).reshape(tar_shape)
            param.data = new_weight
        else:
            dev = param.device
            bias = np.load(os.path.join(directory, name), allow_pickle=True)
            param.data = torch.from_numpy(bias).to(dev)
    return new_state


if __name__ == '__main__':
    path1 = './exp/Deeplabv3+_resnet101.pth'
    model1 = DeepLabv3_plus(in_channels=3, num_classes=15, os=16, pretrained=False, norm_layer=nn.BatchNorm2d,
                            backend="resnet101")
    model1.load_state_dict(torch.load(path1, map_location='cpu'))

    print('compressing the first model...')
    apply_weight_sharing(model1, 8)
    huffman_encode_model(model1, './submit/deeplab_resnet')

    path2 = './exp/DANet_resnet101.pth'
    model2 = DANet(backbone='resnet101', nclass=15, pretrained=False, norm_layer=nn.BatchNorm2d)
    model2.load_state_dict(torch.load(path2, map_location='cpu'))

    print('compressing the second model...')
    apply_weight_sharing(model2, 8)
    huffman_encode_model(model2, './submit/danet')

    path3 = './exp/Deeplabv3+_resnest101.pth'
    model3 = DeepLabv3_plus(in_channels=3, num_classes=15, os=16, pretrained=False, norm_layer=nn.BatchNorm2d,
                            backend="resnest101")
    model3.load_state_dict(torch.load(path3, map_location='cpu'))

    print('compressing the third model...')
    apply_weight_sharing(model3, 8)
    huffman_encode_model(model3, './submit/deeplab_resnest')

    print('compress done')
