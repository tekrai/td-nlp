import csv
import random


def write_to_file(filename: str, head: list, body: list) -> None:
    with open(filename, "w") as file:
        csv_writer = csv.writer(file)
        # Write the header
        csv_writer.writerow(head)
        # Write the rows in group 1
        csv_writer.writerows(body)
    file.close()


with open("data/raw/train.csv", "r", newline='') as f:
    csvReader = csv.reader(f, delimiter=',')

    header = next(csvReader)

    rows = list(csvReader)

    # Shuffle the rows randomly
    random.shuffle(rows)

    # Calculate the split index based on the split ratio
    split_index = int(len(rows) * 0.8)

    # Divide the rows into two groups
    group_1 = rows[:split_index]
    group_2 = rows[split_index:]

    write_to_file("data/raw/train1.csv", header, group_1)
    write_to_file("data/raw/test.csv", header, group_2)
