from sys import argv
import csv
from camino.utils.data import read_json
from os import path

if argv[1] == argv[2]:
    raise Exception("Same arguments")
if path.exists(argv[2]):
    raise Exception("CSV already exists!")

data = read_json(argv[1])
with open(argv[2], "w") as f:
    cf = csv.writer(f, dialect="excel")
    cf.writerows(data['data'])
