# This file is AI generated.
"""
Minimal reader for Microsoft compound files (OLE2 / MS-CFB), read-only.

Compound files hold a small file system (storages and streams) inside one file.
Biacore result files (.blr) use this container. Only what is needed to read every
stream is implemented: the header, the DIFAT, the FAT, the directory tree, and the
mini stream used for small streams. See the [MS-CFB] specification.
"""
import struct

SIGNATURE = b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'

# Special sector numbers
_MAX_REGULAR = 0xFFFFFFFA
_END_OF_CHAIN = 0xFFFFFFFE
_FREE = 0xFFFFFFFF
_NO_STREAM = 0xFFFFFFFF

# Directory entry object types
_STORAGE, _STREAM, _ROOT = 1, 2, 5

__all__ = ['read_compound_file', 'is_compound_file']


def is_compound_file(path):
    """
    Return True if the file starts with the compound file signature.

    Parameters
    ----------
    path : str
        Path to the file.

    Returns
    -------
    bool
        Whether the file is a compound file.
    """
    with open(path, 'rb') as f:
        return f.read(8) == SIGNATURE


def _clean_name(name):
    """Drop the control characters some writers put in front of names (e.g. '\\x03Keywords')."""
    return ''.join(ch for ch in name if ord(ch) >= 32)


def read_compound_file(path):
    """
    Read every stream of a compound file.

    Parameters
    ----------
    path : str
        Path to the compound file.

    Returns
    -------
    dict
        Stream contents (bytes), keyed by their path inside the file, for example
        '_Cycle 1/_Window 1/_Curve 1/Keywords'. Control characters at the start of
        names are removed.

    Raises
    ------
    ValueError
        If the file is not a compound file or its structure is inconsistent.
    """
    with open(path, 'rb') as f:
        data = f.read()

    return _read_compound_bytes(data)


def _read_compound_bytes(data):
    """Parse a compound file held in memory; see read_compound_file."""
    if len(data) < 512 or data[:8] != SIGNATURE:
        raise ValueError("Not a compound file (OLE2 signature missing).")

    (major_version, byte_order, sector_shift, mini_sector_shift) = struct.unpack_from('<HHHH', data, 0x1A)
    if byte_order != 0xFFFE:
        raise ValueError("Unsupported compound file byte order.")
    if major_version not in (3, 4):
        raise ValueError(f"Unsupported compound file version {major_version}.")

    sector_size = 1 << sector_shift
    mini_sector_size = 1 << mini_sector_shift

    (n_fat_sectors, first_dir_sector, _, mini_cutoff, first_mini_fat,
     n_mini_fat, first_difat, n_difat) = struct.unpack_from('<IIIIIIII', data, 0x2C)

    n_sectors = (len(data) - sector_size) // sector_size if len(data) > sector_size else 0

    def sector(index):
        if index > _MAX_REGULAR or index >= n_sectors + 1:
            raise ValueError(f"Sector {index} is outside the file.")
        start = (index + 1) * sector_size
        return data[start:start + sector_size]

    # DIFAT: the first 109 FAT sector numbers are in the header, the rest in DIFAT sectors
    fat_sectors = list(struct.unpack_from('<109I', data, 0x4C))
    difat_sector, entries_per_sector = first_difat, sector_size // 4
    for _ in range(n_difat):
        if difat_sector > _MAX_REGULAR:
            break
        values = struct.unpack(f'<{entries_per_sector}I', sector(difat_sector))
        fat_sectors.extend(values[:-1])
        difat_sector = values[-1]
    fat_sectors = [s for s in fat_sectors if s <= _MAX_REGULAR][:n_fat_sectors]

    fat = []
    for s in fat_sectors:
        fat.extend(struct.unpack(f'<{entries_per_sector}I', sector(s)))

    def chain(start, table, limit):
        """Sector numbers of a chain, guarding against loops."""
        sectors, current = [], start
        while current <= _MAX_REGULAR:
            if current >= len(table) or len(sectors) > limit:
                raise ValueError("Broken sector chain in compound file.")
            sectors.append(current)
            current = table[current]
        return sectors

    def read_chain(start, size=None):
        content = b''.join(sector(s) for s in chain(start, fat, len(fat)))
        return content if size is None else content[:size]

    # Directory entries, 128 bytes each
    directory = read_chain(first_dir_sector)
    entries = []
    for offset in range(0, len(directory) - 127, 128):
        name_length, object_type = struct.unpack_from('<HB', directory, offset + 0x40)
        left, right, child = struct.unpack_from('<III', directory, offset + 0x44)
        start, size = struct.unpack_from('<IQ', directory, offset + 0x74)
        name = directory[offset:offset + max(name_length - 2, 0)].decode('utf-16-le', errors='replace')
        if major_version == 3:
            size &= 0xFFFFFFFF  # the high 32 bits may be garbage in version 3 files
        entries.append(dict(name=name, type=object_type, left=left, right=right, child=child,
                            start=start, size=size))

    if not entries or entries[0]['type'] != _ROOT:
        raise ValueError("Compound file has no root entry.")

    # Small streams live in the mini stream (the root entry's data), in 64-byte mini sectors
    root = entries[0]
    mini_stream = read_chain(root['start'], root['size']) if root['start'] <= _MAX_REGULAR else b''
    mini_fat = []
    if n_mini_fat and first_mini_fat <= _MAX_REGULAR:
        raw = read_chain(first_mini_fat)
        mini_fat = list(struct.unpack(f'<{len(raw) // 4}I', raw))

    def read_stream(entry):
        if entry['size'] == 0:
            return b''
        if entry['size'] < mini_cutoff:
            sectors = chain(entry['start'], mini_fat, len(mini_fat))
            content = b''.join(mini_stream[s * mini_sector_size:(s + 1) * mini_sector_size] for s in sectors)
        else:
            content = read_chain(entry['start'])
        if len(content) < entry['size']:
            raise ValueError(f"Stream '{entry['name']}' is shorter than its declared size.")
        return content[:entry['size']]

    # Walk the tree: each storage's children form a binary tree through left/right siblings
    streams, visited = {}, set()

    def walk(index, prefix):
        pending = [index]
        while pending:
            i = pending.pop()
            if i == _NO_STREAM or i in visited:
                continue
            if i >= len(entries):
                raise ValueError("Directory entry index outside the directory.")
            visited.add(i)
            entry = entries[i]
            pending.extend([entry['left'], entry['right']])
            path = f"{prefix}{_clean_name(entry['name'])}"
            if entry['type'] == _STREAM:
                streams[path] = read_stream(entry)
            elif entry['type'] == _STORAGE:
                walk(entry['child'], path + '/')

    walk(root['child'], '')

    return streams
