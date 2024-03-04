import json
import csv
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="file name.")
    args = parser.parse_args()
    csvfile = open(f'zero_shot_{args.file}.csv', 'r')
    jsonfile = open(f'zero_shot_{args.file}.jsonl', 'w')
    fieldnames = ("prompt", "index")
    reader = csv.DictReader(csvfile, fieldnames)
    for i, row in enumerate(reader):
        if i==0:
            continue
        json.dump(row, jsonfile)
        jsonfile.write('\n')



