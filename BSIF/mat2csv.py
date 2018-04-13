import sys

with open(sys.argv[1]) as input_file:
    for line in input_file:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        fields = line.split(" ")
        comman_joinded = ",".join(fields)
        print(comman_joinded)
