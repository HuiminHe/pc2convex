
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--echo", action='store_true', help="echo the string you use here")
args = parser.parse_args()
print args.echo
