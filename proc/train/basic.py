'''
The file is the basic functions
'''


import numpy as np
import math
import json
from struct import pack
import struct


'''
Input:  
binFileName:    A binary file that stores a list of bit (feature) vectors
                format: 
                1) 64-bit integer: The number of bit vectors.
                2) 32-bit integer: How many btyes needed.
                3) 32-bit integer: How many bits needed.
                4) Others: feature vectors
Output:
Header:         Header of the file
Matrix_         The list of feature vectors (Data)
'''
def readBinFile(binFileName):
    # Read the binary file (uint is byte)
    Matrix_ = np.fromfile(binFileName, dtype = np.uint8)
    # Read the header of the file
    fid = open(binFileName, 'rb')
    N = fid.read(8)
    N = struct.unpack('Q', N)[0]
    X = fid.read(8)
    (D, B) = struct.unpack('II', X)
    A = struct.unpack('Q', X)[0]
    dimensionBits = D
    dimensionBytes = B
    # Delete the header
    Matrix_ = Matrix_[16 : ]
    # reshape the Matrix into list of 128-bit feature vectors
    Matrix_.shape = -1, dimensionBytes
    return N, A, D, Matrix_


'''
Input
binClusterFile: A binary file that stores a D * D std matrix
Output:
Cluster_matrix: The matrix of std
'''
def readClusterFile(binClusterFile):
    Cluster_matrix = np.fromfile(binClusterFile, dtype = np.float32)
    stdNum = len(Cluster_matrix)
    dimensionBits = (int)(math.sqrt(stdNum));
    Cluster_matrix.shape = dimensionBits, -1

    return Cluster_matrix


def ByteTobinary(num):
    f = list(bin(np.uint8(num))[2:])
    c = len(f)
    f = [0] * (8 - c) + f
    return f

'''
The function is to get the binary form of the matrix.
Input:
Matrix:     The matrix to be processed. The unit of the matrix is byte(uint8)
            The matrix is -1, 16

Output:
Matrix_:    The matrix that are binary codes. The uint of matrix is bit 
            The matrix is -1, 128
'''
def unpackBitCode(Matrix):
    M = []
    for mat in Matrix:
        V = []
        for num in mat:
            V = V + ByteTobinary(num)
        M.append(V)
    return np.array(M, dtype = np.uint8)


'''
The function is to shuffle the matrix according to a standard hash map 
that is stored in the file bitMapFile
Input:
bitMapFile:     The file that stores the bit map. The file is JSON format
Matrix:         The matrix to be shuffled. The matrix should be -1, 128 form

Output:
Matrix:         The matrix that has been shuffled. The matrix will be -1, 128 form
'''
def shuffleMatrix(bitMapFile, Matrix):
    bitMap = json.load(open(bitMapFile))
    Matrix_ = np.zeros(Matrix.shape, dtype = np.uint8)
    for k, v in bitMap.items():
        k = (int)(k)
        v = (int)(v)
        Matrix_[: , v] = Matrix[: , k]
    return Matrix_


def cal_offset(value):
    return (int)(value/8), (int)(value%8)

def shuffleMatrixBit(bitMapFile, Matrix):
    bitMap = json.load(open(bitMapFile))
    Matrix_ = np.zeros(Matrix.shape, dtype = np.uint8)
    for k, v in bitMap.items():
        k = (int)(k)
        v = (int)(v)
        order_k, offset_k = cal_offset(k)
        order_v, offset_v = cal_offset(v)
        for i in range(len(Matrix)):
            mask = (Matrix[i][order_k] << offset_k) & np.uint8(128)
            mask = np.uint8(mask >> offset_v)
            Matrix_[i][order_v] = Matrix_[i][order_v] | np.uint8(mask)
    return Matrix_

'''
The function is to transfer a binary form of a matrix
into a decimal form (bytes). (128 -> 16)
Input:
Matrix:         The matrix is a -1, 128 Dimension.

Output:
Matrix_:        The matrix is a -1, 16 Dimension.

'''
def packBitCode(Matrix):
    M = []
    mark = 0
    Value = np.uint8(0)
    for i in range(len(Matrix)):
        fv = []
        mark = 0
        Value = np.uint8(0)
        for j in range(len(Matrix[i])):
            Value = Value | (np.uint8(Matrix[i][j]) << mark)
            # Update mark
            if (j + 1) % 8 == 0 or j == len(Matrix[i]) - 1:
                mark = 0
                fv.append(Value)
                Value = np.uint8(0)
            else:
                mark = mark + 1
        M.append(fv)
    M = np.array(M)
    return M
    
def BinaryToByte(bitlist):
    out = 0
    for bit in bitlist:
        bit = np.uint8(bit)
        out = (out << 1) | bit
    return out

def packBitCode2(Matrix):
    M = []
    for i in range(len(Matrix)):
        fv = []
        for j in range(0, len(Matrix[i]), 8):
            fv.append(BinaryToByte(Matrix[i][j : j + 8]))
        M.append(fv)
    M = np.array(M)
    return M


'''
The function is to transfer a matrix into a binary file
The matrix is byte units based
'''
def writeBinFile(binFileName, N, X, Matrix):
    fid = open(binFileName, 'wb')
    fid.write(struct.pack('Q', N))
    fid.write(struct.pack('Q', X))
    Matrix.shape = 1, -1
    for value in Matrix[0]:
        fid.write(struct.pack('B', value))
    fid.close()

def saveImagesBinaryCodes(binary_codes, binaryCodesFile, M):
    ''' Used for face embedding
        @param: binary_codes: the binary dataset
        @param: binaryCodesFile: the output file of binary dataset
    '''
    fid = open(binaryCodesFile, 'wb')
    fid.write(struct.pack('Q', binary_codes.shape[0]))
    #bitNum = np.uint64(M*8)
    #M = np.uint64(M)
    bitNum = M*8
    X = (M << 32) + bitNum
    #X = (bitNum << 32) + M
    fid.write(struct.pack('Q', X))
    binary_codes.shape = 1, -1
    for value in binary_codes[0]:
        fid.write(struct.pack('B', value))
    fid.close()



''' The function reads .txt strings and save to a list
'''
def readStringArr(string_file):
    data = []
    for line in open(string_file):
        data.append(line)
    return data

''' The function write string list to a file
'''
def writeStringArr(string_arr, string_file):
    f = open(string_file, 'w')
    for str_d in string_arr:
        f.write(str_d)
    f.close()

''' The function write string list of a intervel [) to a file
'''
def writeStringArrInterval(string_arr, start, end, string_file):
    f = open(string_file, 'w')
    for rid in range(start, end, 1):
        f.write(string_arr[rid])
    f.close()


''' The function is to transfer to bag of grams
'''
def stringToBagofgrams(record_s, gram_len):
    record_v = []
    if len(record_s) < gram_len:
        s = ''
        for i in range(gram_len - len(record_s)):
            s += '#'
        record_v.append(record_s + s)
        return
    freq = {}
    for sid in range(len(record_s) - gram_len + 1):
        ss = record_s[sid: sid + gram_len]
        if ss not in freq:
            freq[ss] = 0
            record_v.append(ss)
        else:
            freq[ss] += 1
            record_v.append(ss + "#" + str(freq[ss]))
    return record_v


