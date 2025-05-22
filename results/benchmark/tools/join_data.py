from sys import argv
from os import path
import csv

if len(argv) <= 3:
    print("Usage: output dataset-csv csvfiles")
    exit(0)

csv_out = argv[1]
dataset = argv[2]
other_csvs = argv[3:]
if path.exists(csv_out):
    raise Exception(f"CSV already exists: {csv_out}")

data = {}
with open(dataset, "r") as f:
    cf = csv.reader(f)
    headers = next(cf)[1:]
    for row in cf:
        data[row[0]] = row[1:]

for subset in other_csvs:
    basename = path.basename(subset).split(".")[0]
    with open(subset, "r") as f:
        cf = csv.reader(f)
        prev_len = len(headers)
        headers.extend([f"{basename}.{key}" for key in next(cf)[2:]])
        new_len = len(headers)
        for row in cf:
            name = row[1].split(".")[0]
            if name in data:
                while len(data[name]) < prev_len:
                    data[name].append("NAN")

                data[name].extend(row[2:])
                while len(data[name]) < new_len:
                    data[name].append("NAN")

with open(csv_out, "w") as f:
    cf = csv.writer(f)
    headers = ["name"] + headers
    cf.writerow(headers)
    data_len = len(headers)
    for key, sdata in data.items():
        row = [key] + sdata
        while len(row) < data_len:
            row.append("NAN")

        if len(row) > data_len:
            print(
                f"Data len {len(row)} not equal to header {data_len}"
            )
        else:
            cf.writerow(row)
