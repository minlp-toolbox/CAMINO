from lxml import etree as ET
import re
from sys import argv
import csv
from os import path

default_parser = ET.XMLParser(remove_blank_text=True)


def read_config(config_path):
    """
    Read configuration.

    :param config_path: path
    :return: ET
    """
    # Remove xmlns
    with open(config_path, 'r', encoding='utf-8') as f:
        f.readline()
        content = "<osrl>" + f.read()
    if len(content) < 10:
        return None

    content = re.sub(r" xmlns=[\"'][^']+['\"]", '', content, count=4)
    content = re.sub(r"<\?xml[^>]+>", '', content)
    # Parse xml
    return ET.XML(content, parser=default_parser)


def get_data(name, osrl_file):
    """Get data osr file."""
    if not path.exists(osrl_file):
        return [name, "NAN", "NAN", "NAN", "NAN"]

    tree = read_config(osrl_file)
    try:
        primal = tree.find(
            ".//other[@name='PrimalObjectiveBound']").attrib["value"]
        time_total = float(tree.find(".//time[@type='Total']").text)
        time_setup = float(
            tree.find(".//time[@type='ProblemInitialization']").text)
        dual = tree.find(".//other[@name='TotalNumberOfDualProblems']").attrib["value"]
        nlp = tree.find(".//other[@name='NumberOfNLPProblems']").attrib["value"]
        return [name, primal, str(time_total - time_setup), dual, nlp]
    except Exception:
        return [name, "NAN", "NAN", "NAN", "NAN"]


if len(argv) != 4:
    print(argv)
    print("Usage: python read_shot.py <problem_list> <base_path-output-files> <output_path>")
    exit(1)


with open(argv[1], "r") as f:
    problems = [key.split(".")[0].split(",")[0] for key in f.readlines()]

base_path = argv[2]
output_path = argv[3]
data = [["id", "path", "obj", "calctime", "nr_milp", "nr_nlp"]]
for i, name in enumerate(problems):
    osrl_file = base_path + "/" + name + ".osrl"
    data.append([i] + get_data(name, osrl_file))

with open(output_path, "w") as f:
    writer = csv.writer(f)
    writer.writerows(data)
