# This code is based on oommfdecode.py by Duncan Parkes:
# https://github.com/deparkes/OOMMFTools/blob/master/oommftools/core/oommfdecode.py
# and also ovftools.py by Peyton Murray
# https://github.com/peytondmurray/mx3tools/blob/master/mx3tools/ovftools.py
#
# How to use: 
# place this file in your working folder and use "from mx3import import unpack"
# and then use as unpack("mumax-generated-file.ovf")
#
# works on both text and binary format (prefer the binary! it is much less heavy)
# works on both scalar and vector data (ouput shape is always [znodes, ynodes, xnodes, valuedim] )

import numpy as np
import struct
import pathlib
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor

def i2s(index):
    return f"{index:06}"

def unpack_fobj(f):
    """Takes a file-like object and tries to import as an OVF."""
    try:
        headers = _read_header(f)

        if headers['data_type'][3] == 'Text':
            return _text_decode(f, headers)
        elif headers['data_type'][3] == 'Binary':
            chunk_size = int(headers['data_type'][4])
            return _fast_binary_decode(f, chunk_size, headers, _endianness(f, chunk_size))
        else:
            print(f"Warning: Unknown data type '{data_type_info[3]}' for {f}")
            return None
    
    except Exception as e:
        print(f"An error occurred while unpacking {f}:\n{e}")
        return None

def unpack(path):
    """Takes a string or pathlib.Path object and tries to import as an OVF."""
    path = pathize(path)

    try:
        with path.open('rb') as f:
            return unpack_fobj(f)

    except Exception as e:
        print(f"An error occurred while unpacking {path}:\n{e}")
        return None

def unpack_multiple(path_pattern, start, end, parallelization='threads', max_workers=None):
    
    if parallelization=='threads':
        Executor = ProcessPoolExecutor
    elif parallelization=='processes':
        Executor = ThreadPoolExecutor
    else:
        print(f"Could not import files using parallelization='{parallelization}'. The only allowed approaches are 'threads' and 'processes'")
    
    paths = [(path_pattern + i2s(i) + ".ovf") for i in range(start, end)]
    
    results_list = []
    with Executor(max_workers=max_workers) as executor:
        results_iterator = executor.map(unpack, paths)
        results_list = list(results_iterator)
    
    invalid_indices = [i for i, val in enumerate(results_list) if not isinstance(val, np.ndarray)]
    if invalid_indices:
        print(f"Could not import files with pattern '{path_pattern}' and indices {invalid_indices}")
        return None
    else:
        return np.stack(results_list, axis=0)
    
def fft_multiple(tens):
    return np.fft.rfft(tens, axis=0)[1:]

def pathize(path):
    """Takes a string or pathlib.Path object and return the corresponding pathlib.Path object.

    Parameters
    ----------
    path : str or pathlib.Path
        Input path

    Returns
    -------
    pathlib.Path
        Returns a pathlib.Path object
    """

    if isinstance(path, str):
        return pathlib.Path(path)
    elif isinstance(path, pathlib.PurePath):
        return path
    else:
        raise TypeError(f'Invalid path type: {type(path)}')

def _read_header(fobj, max_header_bytes=2048):
    """Read headers from OVF file object. fobj must be opened in 'rb' mode (read as bytes).

    Parameters
    ----------
    fobj : file
        OVF file to read, must be opened in bytes mode (mode='rb')
    max_header_bytes : int
        max number of bytes allowed to read while reading the header (default: 2048)

    Returns
    -------
    dict
        Dictionary containing the [important] header keys and values
    """

    headers = {'SimTime': -1, 'Iteration': -1, 'Stage': -1, 'MIFSource': ''}
    header_bytes = bytearray()
    continue_reading = True
    # Read until we find the header terminator or exceed a sane limit
    while continue_reading:
        line = fobj.readline()
        if not line:
            raise IOError(f"EOF reached before any data was found")
        
        header_bytes.extend(line)
        if b'Begin: Data' in line:
            continue_reading = False
        if len(header_bytes) > max_header_bytes:
            raise IOError(f"'Begin: Data' not found in the first {max_header_bytes} bytes")
    
    header_text = header_bytes.decode('latin-1')

    for line_unstripped in header_text.splitlines():
        
        line = line_unstripped.strip()
        
        for key in ["xbase",
                    "ybase",
                    "zbase",
                    "xstepsize",
                    "ystepsize",
                    "zstepsize",
                    "xnodes",
                    "ynodes",
                    "znodes",
                    "valuemultiplier",
                    "valuedim"]:
            if key in line:
                headers[key] = float(line.split(': ')[1])

        if 'Total simulation time' in line:
            headers['SimTime'] = float(line.split(':')[-1].strip().split()[0].strip())
        elif 'Iteration' in line:
            headers['Iteration'] = float(line.split(':')[2].split(',')[0].strip())
        # elif 'Stage' in line:
        #     headers['Stage'] = float(line.split(':')[2].split(',')[0].strip())
        elif 'MIF source file' in line:
            headers['MIFSource'] = line.split(':', 2)[2].strip()
        else:
            continue

    headers['data_type'] = line.split()

    return headers

def _byte_decoder(endianness):
    return struct.Struct(endianness)

def _endianness(f, nbytes):
    buffer = f.read(nbytes)

    big_endian = {4: '>f', 8: '>d'}
    little_endian = {4: '<f', 8: '<d'}
    value = {4: 1234567.0, 8: 123456789012345.0}

    if struct.unpack(big_endian[nbytes], buffer)[0] == value[nbytes]:       # Big endian?
        return big_endian[nbytes]
    elif struct.unpack(little_endian[nbytes], buffer)[0] == value[nbytes]:  # Little endian?
        return little_endian[nbytes]
    else:
        raise IOError(f'Cannot decode {nbytes}-byte order mark: ' + hex(buffer))

def _text_decode(f, headers):
    
    arrshape=[int(headers[key]) for key in ['znodes','ynodes','xnodes','valuedim']]
    
    data = np.loadtxt(f, max_rows=np.prod(arrshape[:3])).reshape(arrshape)

    return data*headers.get('valuemultiplier', 1)

def _fast_binary_decode(f, chunk_size, headers, dtype):

    arrshape=[int(headers[key]) for key in ['znodes','ynodes','xnodes','valuedim']]
    ret = np.ndarray(shape=arrshape,
                     dtype=dtype,
                     buffer=f.read(np.prod(arrshape)*chunk_size),
                     offset=0
                     )

    return ret
