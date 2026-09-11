"""Run the Coat support/composition audit."""

from pathlib import Path

from policyshiftlab.coat_audit import audit_coat_files


def main() -> None:
    train_path = Path("data/coat/train.ascii")
    randomized_path = Path("data/coat/test.ascii")

    if not train_path.exists() or not randomized_path.exists():
        raise SystemExit(
            "Coat files not found. Put the original train.ascii and "
            "test.ascii under data/coat/ before running this audit."
        )

    result = audit_coat_files(
        train_path,
        randomized_path,
        positive_threshold=4,
    )

    print(f"matrix shape: {result.n_users} users x {result.n_items} items")
    print(f"train observed ratings:      {result.train_observed}")
    print(f"randomized observed ratings: {result.randomized_observed}")
    print(f"exact pair overlap:          {result.exact_pair_overlap}")
    print()
    print(
        "train-supported users/items: "
        f"{result.train_supported_users}/{result.train_supported_items}"
    )
    print(
        "randomized-supported users/items: "
        f"{result.randomized_supported_users}/"
        f"{result.randomized_supported_items}"
    )
    print(
        "randomized pairs with train user+item support: "
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
    print(
        "randomized matrix observation fraction: "
        f"{result.randomized_observation_fraction:.4f}"
    )


if __name__ == "__main__":
    main()
