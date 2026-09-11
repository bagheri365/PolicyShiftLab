"""Run the Yahoo! R3 support/composition audit."""

from pathlib import Path

from policyshiftlab.yahoo_r3_audit import audit_yahoo_r3_files


def _resolve_paths(base: Path) -> tuple[Path, Path]:
    """Support original Webscope names and common research-copy names."""
    candidates = (
        (
            base / "ydata-ymusic-rating-study-v1_0-train.txt",
            base / "ydata-ymusic-rating-study-v1_0-test.txt",
        ),
        (
            base / "user.txt",
            base / "random.txt",
        ),
    )

    for train_path, randomized_path in candidates:
        if train_path.exists() and randomized_path.exists():
            return train_path, randomized_path

    expected = "\n  or\n".join(
        f"  {train_path}\n  {randomized_path}"
        for train_path, randomized_path in candidates
    )
    raise SystemExit("Yahoo! R3 files not found. Expected:\n" + expected)


def main() -> None:
    base = Path("data/yahoo_r3")
    train_path, randomized_path = _resolve_paths(base)

    print(f"logged file:     {train_path}")
    print(f"randomized file: {randomized_path}")
    print()

    result = audit_yahoo_r3_files(
        train_path,
        randomized_path,
        positive_threshold=4,
    )

    print(f"train observations:      {result.train_observations}")
    print(f"randomized observations: {result.randomized_observations}")
    print()
    print(
        "users train/randomized/shared: "
        f"{result.train_users}/"
        f"{result.randomized_users}/"
        f"{result.shared_users}"
    )
    print(
        "items train/randomized/shared: "
        f"{result.train_items}/"
        f"{result.randomized_items}/"
        f"{result.shared_items}"
    )
    print(f"exact pair overlap: {result.exact_pair_overlap}")
    print(
        "randomized pairs with logged user+item support: "
        f"{result.randomized_pairs_with_train_user_item_support} "
        f"({result.randomized_support_fraction:.3f})"
    )
    print()
    print(f"train rating support:      {result.train_rating_values}")
    print(f"randomized rating support: {result.randomized_rating_values}")
    print(f"train P(rating >= 4):      {result.train_positive_rate:.3f}")
    print(
        "randomized P(rating >= 4): "
        f"{result.randomized_positive_rate:.3f}"
    )
    print()
    print(
        "train user activity min/median/max: "
        f"{result.train_user_activity_min}/"
        f"{result.train_user_activity_median:.1f}/"
        f"{result.train_user_activity_max}"
    )
    print(
        "randomized user activity min/median/max: "
        f"{result.randomized_user_activity_min}/"
        f"{result.randomized_user_activity_median:.1f}/"
        f"{result.randomized_user_activity_max}"
    )
    print(
        "train item activity min/median/max: "
        f"{result.train_item_activity_min}/"
        f"{result.train_item_activity_median:.1f}/"
        f"{result.train_item_activity_max}"
    )
    print(
        "randomized item activity min/median/max: "
        f"{result.randomized_item_activity_min}/"
        f"{result.randomized_item_activity_median:.1f}/"
        f"{result.randomized_item_activity_max}"
    )
    print()
    print(
        "randomized observations per active user constant: "
        f"{result.randomized_user_count_is_constant}"
    )
    print(
        "randomized items per active user: "
        f"{result.randomized_items_per_active_user}"
    )
    print()
    print(
        "sampling_data.txt is intentionally excluded from this benchmark. "
        "Do not treat the randomized subset as a universal target population "
        "until these support/composition diagnostics are interpreted."
    )


if __name__ == "__main__":
    main()
