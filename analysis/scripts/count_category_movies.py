#!/usr/bin/env python3

from movie_collections import DEFAULT_AGGREGATE_PATH, COLLECTION_ORDER, load_collection_ids


def main():
    collections, valid_ids, per_file_stats = load_collection_ids(DEFAULT_AGGREGATE_PATH, with_stats=True)
    print(f"{len(valid_ids)} movies with a non-blank id in {DEFAULT_AGGREGATE_PATH}\n")

    for name in COLLECTION_ORDER:
        print(f"--- {name} ---")
        for filename, min_rank, max_rank, stats in per_file_stats[name]:
            rank_note = f" (rank {min_rank}-{max_rank})" if min_rank is not None or max_rank is not None else ""
            print(
                f"  {filename}{rank_note}: {stats['counted']} counted / {stats['total_rows']} rows "
                f"(no id: {stats['no_id']}, out of rank range: {stats['out_of_rank_range']}, "
                f"not a non-blank id in aggregate.tsv: {stats['not_in_aggregate']})"
            )
        print(f"  => {name}: {len(collections[name])} movies (deduped across files)")
        print()

    union_ids = set().union(*collections.values())

    print("=== summary ===")
    for name in COLLECTION_ORDER:
        print(f"{name}:{' ' * (9 - len(name))}{len(collections[name])}")
    print(f"total unique (popular or prestige or indie): {len(union_ids)}")


if __name__ == "__main__":
    main()
