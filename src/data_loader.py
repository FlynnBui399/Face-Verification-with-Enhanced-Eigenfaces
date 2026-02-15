"""
Data Loader - Đọc dữ liệu từ MXNet RecordIO và InsightFace .bin format.

Module này implement từ đầu (from scratch) bộ đọc file binary RecordIO
mà KHÔNG dùng thư viện mxnet. Chỉ dùng struct + numpy + PIL.

Formats:
  - .idx: Text file, mỗi dòng "index\\toffset\\n"
  - .rec: Binary RecordIO, mỗi record = magic(4B) + lrecord(4B) + data(nB)
  - .bin: Pickle file chứa (list_jpeg_bytes, issame_list) cho eval datasets

Author: Mathematics for AI - Final Project
"""

import os
import struct
import pickle
import numpy as np
from io import BytesIO
from PIL import Image
from tqdm import tqdm

from src.config import (
    CASIA_REC, CASIA_IDX, CASIA_PROPERTY,
    CACHE_DIR, BATCH_SIZE, IMAGE_SIZE, RANDOM_SEED
)

# ==============================================================================
# CONSTANTS - MXNet RecordIO Format
# ==============================================================================
RECORDIO_MAGIC = 0xCED7230A       # Magic number cho RecordIO
IR_HEADER_FORMAT = '<IfQQ'        # flag(uint32), label(float32), id(uint64), id2(uint64)
IR_HEADER_SIZE = struct.calcsize(IR_HEADER_FORMAT)  # = 24 bytes


# ==============================================================================
# RECORDIO READER (FROM SCRATCH - Không dùng mxnet)
# ==============================================================================

def read_idx_file(idx_path):
    """
    Đọc file .idx (text format) để lấy mapping index → byte offset trong .rec.
    
    Format mỗi dòng: "index\\toffset\\n"
    Ví dụ:
        1\\t0
        2\\t4820
        3\\t9432
    
    Returns:
        dict: {index: offset} mapping
    """
    indices = {}
    with open(idx_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                idx = int(parts[0])
                offset = int(parts[1])
                indices[idx] = offset
    return indices


def read_record_at_offset(rec_file, offset):
    """
    Đọc một record từ file .rec tại vị trí offset.
    
    RecordIO Binary Format:
        - 4 bytes: magic number (0xCED7230A, little-endian)
        - 4 bytes: lrecord
            - bits [0:28] = length of data
            - bits [29:31] = cflag (0=complete, 1=start, 2=middle, 3=end)
        - N bytes: data (N = length)
        - Padding: round up to 4-byte boundary
    
    Args:
        rec_file: File object opened in 'rb' mode
        offset: Byte offset in file
    
    Returns:
        bytes: Raw data of the record
    """
    rec_file.seek(offset)
    header_bytes = rec_file.read(8)
    if len(header_bytes) < 8:
        return None
    
    magic, lrecord = struct.unpack('<II', header_bytes)
    
    # Verify magic number (lower 29 bits)
    if (magic & 0x1FFFFFFF) != (RECORDIO_MAGIC & 0x1FFFFFFF):
        raise ValueError(f"Invalid RecordIO magic number at offset {offset}: "
                         f"got 0x{magic:08X}, expected 0x{RECORDIO_MAGIC:08X}")
    
    # Extract cflag and length
    cflag = lrecord >> 29
    length = lrecord & ((1 << 29) - 1)
    
    if cflag != 0:
        # Multi-part record - chỉ hỗ trợ complete records
        raise NotImplementedError(f"Multi-part records not supported (cflag={cflag})")
    
    data = rec_file.read(length)
    
    # Skip padding to 4-byte boundary
    padded_length = ((length + 3) >> 2) << 2
    if padded_length > length:
        rec_file.read(padded_length - length)
    
    return data


def unpack_image_record(data):
    """
    Giải mã một image record từ raw data.
    
    InsightFace IRHeader Format (24 bytes):
        - uint32 flag:   0 = normal image, 1 = header record
        - float32 label: Class label (subject ID)
        - uint64 id:     Record ID
        - uint64 id2:    Additional ID (thường = 0)
    
    Sau IRHeader là JPEG encoded image bytes.
    
    Args:
        data: Raw record bytes
    
    Returns:
        tuple: (label, jpeg_bytes) hoặc None nếu là header record
    """
    if len(data) < IR_HEADER_SIZE:
        return None
    
    flag, label, id1, id2 = struct.unpack(IR_HEADER_FORMAT, data[:IR_HEADER_SIZE])
    
    # flag=0: normal image record
    # flag=2: header/metadata record → skip
    if flag != 0:
        return None
    
    jpeg_bytes = data[IR_HEADER_SIZE:]
    
    # Validate JPEG magic bytes (FFD8)
    if len(jpeg_bytes) < 100 or jpeg_bytes[0:2] != b'\xff\xd8':
        return None
    
    return int(label), jpeg_bytes


def decode_jpeg_to_array(jpeg_bytes, target_size=None, grayscale=True):
    """
    Decode JPEG bytes thành numpy array.
    
    Args:
        jpeg_bytes: Raw JPEG bytes
        target_size: Tuple (width, height) để resize, None = giữ nguyên
        grayscale: True = convert sang grayscale
    
    Returns:
        numpy.ndarray: Image array, shape (H, W) nếu grayscale, (H, W, 3) nếu RGB
    """
    img = Image.open(BytesIO(jpeg_bytes))
    
    if grayscale:
        img = img.convert('L')
    else:
        img = img.convert('RGB')
    
    if target_size is not None:
        img = img.resize((target_size, target_size), Image.BILINEAR)
    
    return np.array(img, dtype=np.float32)


def read_property_file(property_path):
    """
    Đọc file property của dataset.
    
    Format: "num_classes,height,width"
    Ví dụ: "10572,112,112"
    
    Returns:
        tuple: (num_classes, height, width)
    """
    with open(property_path, 'r') as f:
        line = f.read().strip()
    parts = line.split(',')
    return int(parts[0]), int(parts[1]), int(parts[2])


# ==============================================================================
# CASIA-WEBFACE LOADER
# ==============================================================================

def load_casia_webface(n_samples=None, image_size=IMAGE_SIZE, cache=True):
    """
    Load CASIA-WebFace dataset từ RecordIO format.
    
    Pipeline:
        1. Check cache → nếu có thì load từ cache
        2. Đọc .idx file để lấy record offsets
        3. Sample n_samples records (nếu chỉ định)
        4. Đọc từng record từ .rec, decode JPEG, resize, convert grayscale
        5. Lưu cache cho lần sau
    
    Args:
        n_samples: Số lượng ảnh cần load (None = tất cả, khuyến nghị 50000)
        image_size: Kích thước resize (mặc định 64)
        cache: Có lưu cache không
    
    Returns:
        tuple: (images, labels)
            - images: np.ndarray shape (N, image_size, image_size), dtype float64
            - labels: np.ndarray shape (N,), dtype int
    """
    # Check cache
    cache_img_path = os.path.join(CACHE_DIR, f"train_images_{image_size}x{image_size}_n{n_samples}.npy")
    cache_lbl_path = os.path.join(CACHE_DIR, f"train_labels_n{n_samples}.npy")
    
    if cache and os.path.exists(cache_img_path) and os.path.exists(cache_lbl_path):
        print(f"  [Cache] Loading from cache: {cache_img_path}")
        images = np.load(cache_img_path)
        labels = np.load(cache_lbl_path)
        print(f"  [Cache] Loaded {len(images)} images, {len(np.unique(labels))} subjects")
        return images, labels
    
    # Read property file
    num_classes, orig_h, orig_w = read_property_file(CASIA_PROPERTY)
    print(f"  Dataset: {num_classes} classes, original size: {orig_h}x{orig_w}")
    
    # Read index file
    print("  Reading index file...")
    idx_map = read_idx_file(CASIA_IDX)
    all_indices = sorted(idx_map.keys())
    total_records = len(all_indices)
    print(f"  Total records in dataset: {total_records}")
    
    # Sample if needed
    rng = np.random.RandomState(RANDOM_SEED)
    if n_samples is not None and n_samples < total_records:
        selected_indices = rng.choice(all_indices, size=n_samples, replace=False)
        selected_indices = sorted(selected_indices)
        print(f"  Sampling {n_samples} images from {total_records}")
    else:
        selected_indices = all_indices
        n_samples = total_records
    
    # Read records
    images = np.zeros((n_samples, image_size, image_size), dtype=np.float32)
    labels = np.zeros(n_samples, dtype=np.int32)
    
    valid_count = 0
    
    with open(CASIA_REC, 'rb') as rec_file:
        for i, idx in enumerate(tqdm(selected_indices, desc="  Loading images")):
            try:
                offset = idx_map[idx]
                data = read_record_at_offset(rec_file, offset)
                if data is None:
                    continue
                
                result = unpack_image_record(data)
                if result is None:
                    continue
                
                label, jpeg_bytes = result
                
                img = decode_jpeg_to_array(jpeg_bytes, target_size=image_size, grayscale=True)
                
                images[valid_count] = img
                labels[valid_count] = label
                valid_count += 1
                
            except Exception as e:
                continue  # Skip corrupted records
    
    # Trim to valid count
    images = images[:valid_count]
    labels = labels[:valid_count]
    
    print(f"  Successfully loaded {valid_count} images, {len(np.unique(labels))} subjects")
    
    # Save cache
    if cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(cache_img_path, images)
        np.save(cache_lbl_path, labels)
        print(f"  [Cache] Saved to {CACHE_DIR}")
    
    return images, labels


# ==============================================================================
# EVAL DATASET LOADER (.bin format)
# ==============================================================================

def load_eval_bin(bin_path, image_size=IMAGE_SIZE):
    """
    Load evaluation dataset từ InsightFace .bin format.
    
    .bin file format (pickle):
        - Tuple: (bins, issame_list)
        - bins: list of JPEG byte strings
            - bins[2*i] và bins[2*i+1] tạo thành pair i
        - issame_list: list of booleans
            - issame_list[i] = True nếu pair i cùng người
    
    Standard protocols:
        - LFW: 6000 pairs (12000 images)
        - CFP-FP: 7000 pairs (14000 images)  
        - AgeDB-30: 6000 pairs (12000 images)
    
    Args:
        bin_path: Path to .bin file
        image_size: Target resize dimension
    
    Returns:
        tuple: (images, issame_list)
            - images: np.ndarray shape (N, image_size, image_size), dtype float64
            - issame_list: np.ndarray of booleans, shape (N/2,)
    """
    cache_name = os.path.splitext(os.path.basename(bin_path))[0]
    cache_img_path = os.path.join(CACHE_DIR, f"eval_{cache_name}_{image_size}x{image_size}.npy")
    cache_lbl_path = os.path.join(CACHE_DIR, f"eval_{cache_name}_issame.npy")
    
    if os.path.exists(cache_img_path) and os.path.exists(cache_lbl_path):
        print(f"  [Cache] Loading {cache_name} from cache")
        images = np.load(cache_img_path)
        issame = np.load(cache_lbl_path)
        return images, issame
    
    print(f"  Loading {bin_path}...")
    
    with open(bin_path, 'rb') as f:
        # Try different pickle encodings
        try:
            bins, issame_list = pickle.load(f, encoding='bytes')
        except (UnicodeDecodeError, TypeError):
            f.seek(0)
            bins, issame_list = pickle.load(f)
    
    n_images = len(bins)
    issame = np.array(issame_list, dtype=bool)
    
    print(f"  {cache_name}: {n_images} images, {len(issame)} pairs")
    
    images = np.zeros((n_images, image_size, image_size), dtype=np.float32)
    
    for i in tqdm(range(n_images), desc=f"  Decoding {cache_name}"):
        try:
            img = decode_jpeg_to_array(bins[i], target_size=image_size, grayscale=True)
            images[i] = img
        except Exception as e:
            # Nếu decode lỗi, giữ ảnh đen (zeros)
            pass
    
    # Save cache
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(cache_img_path, images)
    np.save(cache_lbl_path, issame)
    print(f"  [Cache] Saved {cache_name} to cache")
    
    return images, issame


def load_all_eval_datasets(image_size=IMAGE_SIZE):
    """
    Load tất cả eval datasets.
    
    Returns:
        dict: {name: (images, issame_list)} for each dataset
    """
    from src.config import LFW_BIN, CFP_FP_BIN, AGEDB_BIN
    
    eval_datasets = {}
    
    datasets = [
        ("LFW", LFW_BIN),
        ("CFP-FP", CFP_FP_BIN),
        ("AgeDB-30", AGEDB_BIN),
    ]
    
    for name, path in datasets:
        if os.path.exists(path):
            print(f"\n  Loading {name}...")
            images, issame = load_eval_bin(path, image_size)
            eval_datasets[name] = (images, issame)
        else:
            print(f"  [Warning] {name} not found at {path}, skipping")
    
    return eval_datasets
