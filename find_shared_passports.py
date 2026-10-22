
from datetime import datetime
from pathlib import Path
import argparse

import pandas as pd


INPUT_FILE = Path(__file__).resolve().parent / "cleaned_v2" / "cleaned_v2" / "users.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "shared_passport_profiles"


def find_profiles(users):
    if users["user_id"].eq("").any() or users["user_id"].duplicated().any():
        raise ValueError("users.csv must have nonempty unique user_id values")
    # Сравниваем готовые ключи паспорта.
    candidates = users.loc[users["passport_key"].ne("")].copy()
    counts = candidates.groupby("passport_key")["user_id"].nunique()
    counts = counts.loc[counts.gt(1)]
    matches = candidates.loc[candidates["passport_key"].isin(counts.index)].copy()
    matches["passport_group"] = [f"G{i+1:04d}" for i in pd.factorize(matches["passport_key"], sort=True)[0]]
    matches["profiles_on_passport"] = matches["passport_key"].map(counts)
    # Если полного ФИО нет, собираем из частей.
    fallback = (matches["last_name"] + " " + matches["first_name"] + " " + matches["middle_name"])
    fallback = fallback.str.replace(r"\s+", " ", regex=True).str.strip()
    matches["display_name"] = matches["full_name"].where(matches["full_name"].ne(""), fallback)
    fields = ["passport_group", "passport_key", "user_id", "display_name", "full_name", "first_name", "middle_name",
              "last_name", "birth_date_iso", "sex", "profiles_on_passport", "identity_status", "user_needs_review"]
    return matches.sort_values(["passport_group", "user_id"])[fields]


def run(input_file, output_root):
    columns = ["user_id", "passport_key", "full_name", "first_name", "middle_name", "last_name", "birth_date_iso", "sex",
               "identity_status", "user_needs_review"]
    users = pd.read_csv(input_file, sep=";", encoding="utf-8-sig", dtype=str,
                        keep_default_na=False, usecols=columns)
    matches = find_profiles(users)
    output = output_root / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    result = output / "profiles_with_shared_passports.csv"
    matches.to_csv(result, sep=";", encoding="utf-8-sig", index=False)
    print(f"Групп с общим паспортом: {matches['passport_group'].nunique()}")
    print(f"Профилей: {len(matches)}")
    print(f"Файл: {result}")
    print("Готово.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, default=INPUT_FILE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    run(args.input_file, args.output_dir)


if __name__ == "__main__":
    main()

