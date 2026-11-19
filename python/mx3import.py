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
from concurrent.futures import ThreadPoolExecutor

def i2s(index):
    return f"{index:06}"

class OVFUnpackError(Exception):
    """Custom exception for specific OVF parsing failures."""
    pass

def unpack_fobj(f, with_header=False, use_valuemultiplier=False):
    """Takes a file-like object and tries to import as an OVF.
    With with_header=True returns (data, headers)."""
    try:
        headers = _read_header(f)

        if headers['data_type'][3] == 'Text':
            data = _text_decode(f, headers, use_valuemultiplier)
        elif headers['data_type'][3] == 'Binary':
            chunk_size = int(headers['data_type'][4])
            data = _fast_binary_decode(f, chunk_size, headers, _endianness(f, chunk_size), use_valuemultiplier)
        else:
            raise OVFUnpackError(f"Unknown data type '{headers['data_type'][3]}'")
    
    except Exception as e:
        raise OVFUnpackError(f"Failed to decode file stream.") from e

    return (data, headers) if with_header else data

def unpack(path, with_header=False, use_valuemultiplier=False):
    """Takes a string or pathlib.Path object and tries to import as an OVF.
    With with_header=True returns (data, headers)."""
    path = pathize(path)

    try:
        with path.open('rb') as f:
            return unpack_fobj(f, with_header, use_valuemultiplier)

    except Exception as e:
        print(f"An error occurred while unpacking {path}:\n{e}")
        raise e

def unpack_multiple(path_pattern, start, end, slicer_tuple=None, max_workers=None, with_headers=False):
    """With with_headers=True returns (data, list of headers)."""
    if slicer_tuple is None:
        slicer_tuple = slice(None)
    try:
        paths = [(path_pattern + i2s(i) + ".ovf") for i in range(start, end)]
    
        first, first_headers = unpack(paths[0], with_header=True)
        first = first[slicer_tuple]
        result_arr = np.empty_like(first, shape=[end-start, *(first.shape)]) 
        result_arr[0] = first
        headers = [first_headers] + [None] * (end-start-1)
        
        def load_file(indexed_path):
            idxx, pathh = indexed_path
            try:
                data, headers[idxx] = unpack(pathh, with_header=True)
                result_arr[idxx] = data[slicer_tuple]
            except Exception as e:
                print(f"Error processing index {idxx} ({pathh}): {e}")
                raise e

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # list() collects the results, so an exception in a worker is re-raised here
            list(executor.map(load_file, list(enumerate(paths))[1:]))
    
        return (result_arr, headers) if with_headers else result_arr
    except Exception as inst:
        print(type(inst))   
        print(inst.args)     
        print(inst)       
        raise inst

import xarray as xr
def unpack_into_xarray(path, absolute_coords=False, use_valuemultiplier=False):
    """Takes a string or pathlib.Path object and tries to import as an OVF.
    With absolute_coords=True, the coordinates are the cell centres as in the file
    (base + i*stepsize), otherwise they are centered on 0."""
    
    path = pathize(path)
    with path.open('rb') as f:
        data, headers = unpack_fobj(f, with_header=True, use_valuemultiplier=use_valuemultiplier)

    nx = int(headers.get("xnodes", data.shape[2]))
    ny = int(headers.get("ynodes", data.shape[1]))
    nz = int(headers.get("znodes", data.shape[0]))
    valuedim = int(headers.get("valuedim", data.shape[3] if data.ndim == 4 else 1))

    dx = headers.get("xstepsize", 1.0)
    dy = headers.get("ystepsize", 1.0)
    dz = headers.get("zstepsize", 1.0)

    if absolute_coords:
        # cell centres: base + i*step
        x = headers.get("xbase", dx / 2) + np.arange(nx) * dx
        y = headers.get("ybase", dy / 2) + np.arange(ny) * dy
        z = headers.get("zbase", dz / 2) + np.arange(nz) * dz
    else:
        # centered coordinates
        x = (np.arange(nx) - (nx - 1) / 2) * dx
        y = (np.arange(ny) - (ny - 1) / 2) * dy
        z = (np.arange(nz) - (nz - 1) / 2) * dz

    # Component labels (vector fields)
    if valuedim == 1:
        comps = ["scalar"]
    else:
        comps = headers.get("valuelabels", "comp_1 comp_2 comp_3").split()
        if len(comps) != valuedim:
            comps = [f"comp_{i}" for i in range(valuedim)]

    meshunit = {"units": headers.get("meshunit", "")}
    coords = {
        "z": ("z", z, meshunit),
        "y": ("y", y, meshunit),
        "x": ("x", x, meshunit),
        "comp": comps
    }

    attrs = {k: v for k, v in headers.items() if isinstance(v, (int, float, str))}
    return xr.DataArray(data, dims=("z", "y", "x", "comp"), coords=coords, name=headers.get("title"), attrs=attrs)

def unpack_multiple_into_xarray(path_pattern, start, end, max_workers=None, absolute_coords=False):
    """Like unpack_multiple, but returns an xarray.DataArray with dims (t, z, y, x, comp),
    t being the 'Total simulation time' of each file."""
    import xarray as xr
    data, headers = unpack_multiple(path_pattern, start, end, max_workers=max_workers, with_headers=True)
    # reads the first file once more, only to reuse the coordinates and attrs of unpack_into_xarray
    template = unpack_into_xarray(path_pattern + i2s(start) + ".ovf", absolute_coords=absolute_coords)
    attrs = {k: v for k, v in template.attrs.items() if k != 'SimTime'}
    t = np.array([h['SimTime'] for h in headers])
    return xr.DataArray(data, dims=("t", *template.dims), coords={"t": ("t", t, {"units": "s"}), **template.coords},
                        name=template.name, attrs=attrs)

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

def _read_header(fobj, max_header_bytes=65536):
    """Read headers from OVF file object. fobj must be opened in 'rb' mode (read as bytes).

    Parameters
    ----------
    fobj : file
        OVF file to read, must be opened in bytes mode (mode='rb')
    max_header_bytes : int
        max number of bytes allowed to read while reading the header (default: 65536)

    Returns
    -------
    dict
        Dictionary containing the [important] header keys and values
    """

    # OVF 1.0 has no valuedim line: it is always a 3-vector field
    headers = {'SimTime': -1, 'Iteration': -1, 'Stage': -1, 'MIFSource': '', 'valuedim': 3}
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
        name = line.lstrip('#').split(':')[0].strip()
        
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
            if key == name:
                headers[key] = float(line.split(': ')[1])

        for key in ["title",
                    "meshunit",
                    "valuelabels",
                    "valueunits"]:
            if key == name.lower():
                headers[key] = line.split(':', 1)[1].strip()

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
        raise IOError(f'Cannot decode {nbytes}-byte order mark: ' + buffer.hex())

def _text_decode(f, headers, use_valuemultiplier=False):
    
    arrshape=[int(headers[key]) for key in ['znodes','ynodes','xnodes','valuedim']]
    
    data = np.loadtxt(f, max_rows=np.prod(arrshape[:3])).reshape(arrshape)

    if use_valuemultiplier:
        data *= headers.get('valuemultiplier', 1)

    return data

def _fast_binary_decode(f, chunk_size, headers, dtype, use_valuemultiplier=False):

    arrshape=[int(headers[key]) for key in ['znodes','ynodes','xnodes','valuedim']]
    buffer = bytearray(int(np.prod(arrshape))*chunk_size)
    if f.readinto(buffer) != len(buffer):
        raise IOError(f"File is truncated: data block is shorter than the header says")
    ret = np.ndarray(shape=arrshape,
                     dtype=dtype,
                     buffer=buffer,
                     offset=0
                     )
    # native byte order ('<f' -> float32)
    ret = ret.astype(ret.dtype.newbyteorder('='), copy=False)

    if use_valuemultiplier and ('valuemultiplier' in headers):
        ret *= headers['valuemultiplier']

    return ret
