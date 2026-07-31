import argparse

from check import check_cache, check_labels


def main():
    parser = argparse.ArgumentParser(description="Run checks on the feature dataset cache and manifest.")
    parser.add_argument("--cache", action="store_true", help="Check the feature dataset cache and manifest for consistency.")
    parser.add_argument("--labels", action="store_true", help="Check the labels in the feature dataset manifest for potential mislabelling.")
    args = parser.parse_args()

    if args.cache:
        print("Running cache and manifest check...")
        exit_code = check_cache()
        if exit_code != 0:
            print("Cache and manifest check failed.")
            return exit_code
        else:
            print("Cache and manifest check passed.")

    if args.labels:
        print("Running label check...")
        exit_code = check_labels()
        if exit_code != 0:
            print("Label check failed.")
            return exit_code
        else:
            print("Label check passed.")

    if not args.cache and not args.labels:
        print("No checks specified. Use --cache or --labels to run checks.")
        return 1

    return 0

if __name__ == "__main__":
    SystemExit(main())