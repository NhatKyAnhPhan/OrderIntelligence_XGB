"""Chia lai cac file parquet thanh file 250.000 dong, dat ten part_0400, part_0401, ...

Chay:
  python rechunk.py --input D:/data/old_20_parts --output D:/data/ecommerce_100m
"""
import argparse
import glob
import os

import pyarrow as pa
import pyarrow.parquet as pq

CHUNK = 250_000  # so dong moi file theo quy uoc nhom
START = 400      # file dau tien cua Nam: part_0400

p = argparse.ArgumentParser()
p.add_argument("--input", required=True, help="thu muc chua cac file parquet cu")
p.add_argument("--output", required=True, help="thu muc moi (khac thu muc input)")
a = p.parse_args()

os.makedirs(a.output, exist_ok=True)
files = sorted(glob.glob(os.path.join(a.input, "*.parquet")))


def write(table, n):
    pq.write_table(table, os.path.join(a.output, f"part_{n:04d}.parquet"), compression="snappy")


buf, n, total = None, START, 0
for f in files:
    t = pq.read_table(f)
    buf = t if buf is None else pa.concat_tables([buf, t])
    while buf.num_rows >= CHUNK:
        write(buf.slice(0, CHUNK), n)
        total += CHUNK
        buf = buf.slice(CHUNK)
        n += 1
    print(f"{os.path.basename(f)} xong -> den part_{n - 1:04d}")

if buf is not None and buf.num_rows > 0:  # phan du neu tong khong chia het cho 250k
    write(buf, n)
    total += buf.num_rows
    print(f"CANH BAO: con {buf.num_rows} dong du, da ghi vao part_{n:04d}")
    n += 1

print(f"Tong: {n - START} file, {total:,} dong (part_{START:04d} -> part_{n - 1:04d})")