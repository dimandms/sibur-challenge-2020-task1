import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="""Pipeline flags""")
    parser.add_argument("--verbose", default=False,
                        action="store_true", help="Show verbose results")
    args = parser.parse_args()

    return args
