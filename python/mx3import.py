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

class OVFUnpackError(Exception):
    """Custom exception for specific OVF parsing failures."""
    pass

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
            raise OVFUnpackError(f"Unknown data type '{data_type_info[3]}'")
    
    except Exception as e:
        raise OVFUnpackError(f"Failed to decode file stream.") from e

def unpack(path):
    """Takes a string or pathlib.Path object and tries to import as an OVF."""
    path = pathize(path)

    try:
        with path.open('rb') as f:
            return unpack_fobj(f)

    except Exception as e:
        print(f"An error occurred while unpacking {path}:\n{e}")
        raise e

def unpack_multiple(path_pattern, start, end, parallelization='threads', max_workers=None):
    
    if parallelization=='threads':
        Executor = ThreadPoolExecutor
    elif parallelization=='processes':
        Executor = ProcessPoolExecutor
    else:
        print(f"Could not import files using parallelization='{parallelization}'. The only allowed approaches are 'threads' and 'processes'")
        return None
    
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

def unpack_preallocate(path_pattern, start, end, slicer_tuple=None):
    if slicer_tuple is None:
        slicer_tuple=slice(None)
    try:
        paths = [(path_pattern + i2s(i) + ".ovf") for i in range(start, end)]
    
        first = unpack(paths[0])[slicer_tuple]    
        result_arr = np.empty_like(first, shape=[end-start, *(first.shape)] ) 
         
        for idxx, pathh in enumerate(paths):
            result_arr[idxx] = unpack(pathh)[slicer_tuple]  
    
        return result_arr
    except Exception as inst:
        print(type(inst))   
        print(inst.args)     
        print(inst)       
        return None

def unpack_preallocate_threaded(path_pattern, start, end, slicer_tuple=None, max_workers=None):
    if slicer_tuple is None:
        slicer_tuple = slice(None)
    try:
        paths = [(path_pattern + i2s(i) + ".ovf") for i in range(start, end)]
    
        first = unpack(paths[0])[slicer_tuple]    
        result_arr = np.empty_like(first, shape=[end-start, *(first.shape)]) 
        
        def load_file(indexed_path):
            idxx, pathh = indexed_path
            try:
                result_arr[idxx] = unpack(pathh)[slicer_tuple]
            except Exception as e:
                print(f"Error processing index {idxx} ({pathh}): {e}")
                raise e

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(load_file, enumerate(paths))
    
        return result_arr
    except Exception as inst:
        print(type(inst))   
        print(inst.args)     
        print(inst)       
        return None

import xarray as xr
def unpack_into_xarray(path):
   """Takes a string or pathlib.Path object and tries to import as an OVF."""
    path = pathize(path)
    with path.open('rb') as f:
        headers = _read_header(f)
        f.seek(0)
        data = unpack_fobj(f)

    nx = int(headers.get("xnodes", data.shape[2]))
    ny = int(headers.get("ynodes", data.shape[1]))
    nz = int(headers.get("znodes", data.shape[0]))
    valuedim = int(headers.get("valuedim", data.shape[3] if data.ndim == 4 else 1))

    dx = headers.get("xstepsize", 1.0)
    dy = headers.get("ystepsize", 1.0)
    dz = headers.get("zstepsize", 1.0)

    # centered coordinates
    x = (np.arange(nx) - (nx - 1) / 2) * dx
    y = (np.arange(ny) - (ny - 1) / 2) * dy
    z = (np.arange(nz) - (nz - 1) / 2) * dz

    # Component labels (vector fields)
    if valuedim == 1:
        comps = ["scalar"]
    elif valuedim == 3:
        comps = headers.get("valuelabels", "comp_1 comp_2 comp_3").split()
        if len(comps) != valuedim:
            comps = [f"comp_{i}" for i in range(valuedim)]

    coords = {
        "z": z,
        "y": y,
        "x": x,
        "comp": comps
    }

    # attrs = {k: v for k, v in headers.items() if isinstance(v, (int, float, str))}
    return xr.DataArray(data, coords=coords) #, attrs=attrs)

def fft_multiple(tens):
    return np.fft.rfft(tens, axis=0)[1:]

def fft_multiple_(tens):
    return np.fft.rfft(tens, axis=0)

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
