import argparse

from spd.utils.cli_utils import add_bool_flag

parser = argparse.ArgumentParser()

# parser.add_argument(
# 	"--local",
# 	action=argparse.BooleanOptionalAction,
# 	# type=bool,
# 	# default=False,
# )

add_bool_flag(
    parser,
    "local",
)

# args = parser.parse_args()

# print(f"{parser = }")
# print(f"{args = }")
# print(f"{args.local = }")

print(f"{parser.parse_args(['--local']) = }")
print(f"{parser.parse_args(['--local=True']) = }")
print(f"{parser.parse_args(['--local=False']) = }")
print(f"{parser.parse_args(['--local=1']) = }")
print(f"{parser.parse_args(['--local=0']) = }")
print(f"{parser.parse_args(['--local', 'True']) = }")
print(f"{parser.parse_args(['--local', 'False']) = }")
print(f"{parser.parse_args(['--local', 'yes']) = }")
print(f"{parser.parse_args(['--local', 'no']) = }")
print(f"{parser.parse_args(['--local', '1']) = }")
print(f"{parser.parse_args(['--local', '0']) = }")
print(f"{parser.parse_args(['--no-local']) = }")
