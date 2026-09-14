"""
Generates a small TPC-H dataset for the Python engine, with no dependencies.

The shipped fixtures under ``tests/fixtures/optimizer/tpc-h`` are what the test
suite runs on; this is for feeding the engine locally at a size of your
choosing, or for regenerating that data. Output matches the fixtures: one
``|``-delimited, header-carrying ``<table>.csv.gz`` per table, with the column
order of ``TPCH_SCHEMA``.

Values follow the domains dbgen draws from - the brands, containers, part types,
market segments, order priorities, ship modes and the 1992-1998 order window
that the 22 queries filter on - so the queries select real rows rather than
matching nothing. It is *not* dbgen: the distributions are uniform and the text
columns are synthetic, so results won't match an official run. What it gives you
is data of the right shape, deterministic for a given ``--seed``.

A few queries (q18, q20, q22) are selective enough to come back empty at these
sizes - the shipped fixtures do the same - so raise ``--scale`` if you want them
to select something.

    python -m benchmarks.tpch_datagen --scale 0.01 --out /tmp/tpch

Then point the engine at it:

    python -m benchmarks.tpch_datagen --scale 0.01 --out /tmp/tpch --run
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import os
import random
import typing as t

from tests.helpers import TPCH_SCHEMA

NATIONS = [
    ("ALGERIA", 0), ("ARGENTINA", 1), ("BRAZIL", 1), ("CANADA", 1),
    ("EGYPT", 4), ("ETHIOPIA", 0), ("FRANCE", 3), ("GERMANY", 3),
    ("INDIA", 2), ("INDONESIA", 2), ("IRAN", 4), ("IRAQ", 4),
    ("JAPAN", 2), ("JORDAN", 4), ("KENYA", 0), ("MOROCCO", 0),
    ("MOZAMBIQUE", 0), ("PERU", 1), ("CHINA", 2), ("ROMANIA", 3),
    ("SAUDI ARABIA", 4), ("VIETNAM", 2), ("RUSSIA", 3),
    ("UNITED KINGDOM", 3), ("UNITED STATES", 1),
]  # fmt: skip

REGIONS = ["AFRICA", "AMERICA", "ASIA", "EUROPE", "MIDDLE EAST"]

COLORS = [
    "almond", "antique", "aquamarine", "azure", "beige", "bisque", "black",
    "blanched", "blue", "blush", "brown", "burlywood", "burnished", "chartreuse",
    "chiffon", "chocolate", "coral", "cornflower", "cornsilk", "cream", "cyan",
    "dark", "deep", "dim", "dodger", "drab", "firebrick", "floral", "forest",
    "frosted", "gainsboro", "ghost", "goldenrod", "green", "grey", "honeydew",
    "hot", "indian", "ivory", "khaki", "lace", "lavender", "lawn", "lemon",
    "light", "lime", "linen", "magenta", "maroon", "medium", "metallic",
    "midnight", "mint", "misty", "moccasin", "navajo", "navy", "olive", "orange",
    "orchid", "pale", "papaya", "peach", "peru", "pink", "plum", "powder",
    "puff", "purple", "red", "rose", "rosy", "royal", "saddle", "salmon",
    "sandy", "seashell", "sienna", "sky", "slate", "smoke", "snow", "spring",
    "steel", "tan", "thistle", "tomato", "turquoise", "violet", "wheat", "white",
    "yellow",
]  # fmt: skip

TYPE_HEAD = ["STANDARD", "SMALL", "MEDIUM", "LARGE", "ECONOMY", "PROMO"]
TYPE_MID = ["ANODIZED", "BURNISHED", "PLATED", "POLISHED", "BRUSHED"]
TYPE_TAIL = ["TIN", "NICKEL", "BRASS", "STEEL", "COPPER"]
CONTAINER_HEAD = ["SM", "LG", "MED", "JUMBO", "WRAP"]
CONTAINER_TAIL = ["CASE", "BOX", "BAG", "JAR", "PKG", "PACK", "CAN", "DRUM"]
SEGMENTS = ["AUTOMOBILE", "BUILDING", "FURNITURE", "MACHINERY", "HOUSEHOLD"]
PRIORITIES = ["1-URGENT", "2-HIGH", "3-MEDIUM", "4-NOT SPECIFIED", "5-LOW"]
SHIP_MODES = ["REG AIR", "AIR", "RAIL", "SHIP", "TRUCK", "MAIL", "FOB"]
INSTRUCTIONS = ["DELIVER IN PERSON", "COLLECT COD", "NONE", "TAKE BACK RETURN"]

NOUNS = [
    "deposits", "packages", "requests", "accounts", "foxes", "theodolites",
    "instructions", "dependencies", "excuses", "asymptotes", "platelets",
    "warthogs", "pinto beans", "ideas", "dolphins", "frays",
]  # fmt: skip
ADVERBS = [
    "carefully", "furiously", "slyly", "blithely", "quickly", "boldly",
    "silently", "quietly", "express", "regular", "final", "pending", "special",
]  # fmt: skip

# The order window every date-filtering query is written against.
START_DATE = datetime.date(1992, 1, 1)
END_DATE = datetime.date(1998, 8, 2)

# dbgen's base row counts, at scale factor 1.
SF1_PART = 200_000
SF1_SUPPLIER = 10_000
SF1_CUSTOMER = 150_000
SF1_ORDERS = 1_500_000

# Column names and order come from the schema the queries are written against,
# so the CSVs line up with what the engine and duckdb are both told to expect.
TABLE_COLUMNS = {table: list(columns) for table, columns in TPCH_SCHEMA.items()}


class Generator:
    def __init__(self, scale: float, seed: int) -> None:
        self.random = random.Random(seed)
        self.parts = max(1, round(SF1_PART * scale))
        self.suppliers = max(1, round(SF1_SUPPLIER * scale))
        self.customers = max(1, round(SF1_CUSTOMER * scale))
        self.orders = max(1, round(SF1_ORDERS * scale))

    def text(self, words: int) -> str:
        pick = self.random.choice
        return " ".join(
            pick(ADVERBS) if self.random.random() < 0.5 else pick(NOUNS) for _ in range(words)
        )

    def phone(self, nationkey: int) -> str:
        digits = self.random.randrange
        return f"{10 + nationkey:02}-{digits(100, 1000)}-{digits(100, 1000)}-{digits(1000, 10000)}"

    def date(self) -> datetime.date:
        return START_DATE + datetime.timedelta(
            days=self.random.randrange((END_DATE - START_DATE).days)
        )

    def region(self) -> t.Iterator[tuple]:
        for key, name in enumerate(REGIONS):
            yield key, name, self.text(8)

    def nation(self) -> t.Iterator[tuple]:
        for key, (name, regionkey) in enumerate(NATIONS):
            yield key, name, regionkey, self.text(8)

    def part(self) -> t.Iterator[tuple]:
        for key in range(1, self.parts + 1):
            name = " ".join(self.random.sample(COLORS, 5))
            manufacturer = self.random.randrange(1, 6)
            brand = self.random.randrange(1, 6)
            yield (
                key,
                name,
                f"Manufacturer#{manufacturer}",
                f"Brand#{manufacturer}{brand}",
                f"{self.random.choice(TYPE_HEAD)} {self.random.choice(TYPE_MID)}"
                f" {self.random.choice(TYPE_TAIL)}",
                self.random.randrange(1, 51),
                f"{self.random.choice(CONTAINER_HEAD)} {self.random.choice(CONTAINER_TAIL)}",
                round(900 + (key % 2001) / 10 + key / 1000, 2),
                self.text(5),
            )

    def supplier(self) -> t.Iterator[tuple]:
        # q16 counts suppliers whose comment carries a Customer/Complaints
        # notice and q2 the rest, so both phrasings have to occur.
        for key in range(1, self.suppliers + 1):
            nationkey = self.random.randrange(len(NATIONS))
            comment = self.text(9)

            if key % 47 == 0:
                comment = f"{comment} Customer {self.text(2)} Complaints"
            elif key % 53 == 0:
                comment = f"{comment} Customer {self.text(2)} Recommends"

            yield (
                key,
                f"Supplier#{key:09}",
                self.text(3),
                nationkey,
                self.phone(nationkey),
                round(self.random.uniform(-999.99, 9999.99), 2),
                comment,
            )

    def partsupp(self) -> t.Iterator[tuple]:
        for partkey in range(1, self.parts + 1):
            for i in range(4):
                suppkey = 1 + (partkey + i * (self.suppliers // 4 + 1)) % self.suppliers
                yield (
                    partkey,
                    suppkey,
                    self.random.randrange(1, 10000),
                    round(self.random.uniform(1, 1000), 2),
                    self.text(12),
                )

    def customer(self) -> t.Iterator[tuple]:
        for key in range(1, self.customers + 1):
            nationkey = self.random.randrange(len(NATIONS))
            yield (
                key,
                f"Customer#{key:09}",
                self.text(3),
                nationkey,
                self.phone(nationkey),
                round(self.random.uniform(-999.99, 9999.99), 2),
                self.random.choice(SEGMENTS),
                self.text(10),
            )

    def orders_and_lineitem(self) -> tuple[list[tuple], list[tuple]]:
        orders: list[tuple] = []
        lineitem: list[tuple] = []

        for key in range(1, self.orders + 1):
            orderdate = self.date()
            total = 0.0
            lines = []

            # dbgen gives every order 1 to 7 lines; q18 looks for orders whose
            # quantities total over 300, which needs the upper end of that.
            for line in range(1, self.random.randrange(2, 9)):
                partkey = self.random.randrange(1, self.parts + 1)
                suppkey = self.random.randrange(1, self.suppliers + 1)
                quantity = float(self.random.randrange(1, 51))
                price = round(quantity * self.random.uniform(900, 2000) / 10, 2)
                discount = round(self.random.randrange(0, 11) / 100, 2)
                tax = round(self.random.randrange(0, 9) / 100, 2)
                shipdate = orderdate + datetime.timedelta(days=self.random.randrange(1, 122))
                commitdate = orderdate + datetime.timedelta(days=self.random.randrange(30, 91))
                receiptdate = shipdate + datetime.timedelta(days=self.random.randrange(1, 31))
                # dbgen ties these flags to the receipt date; q1 and q10 both
                # read them, and q12 needs receipts to fall after shipment.
                returnflag = "R" if receiptdate <= END_DATE else "N"
                if returnflag == "R":
                    returnflag = self.random.choice(["R", "A"])
                linestatus = "F" if shipdate <= datetime.date(1995, 6, 17) else "O"

                total += price * (1 - discount) * (1 + tax)
                lines.append(
                    (
                        key,
                        partkey,
                        suppkey,
                        line,
                        quantity,
                        price,
                        discount,
                        tax,
                        returnflag,
                        linestatus,
                        shipdate.isoformat(),
                        commitdate.isoformat(),
                        receiptdate.isoformat(),
                        self.random.choice(INSTRUCTIONS),
                        self.random.choice(SHIP_MODES),
                        self.text(4),
                    )  # fmt: skip
                )

            comment = self.text(6)

            # q13 counts orders whose comment does *not* match
            # '%special%requests%', so some have to match.
            if key % 31 == 0:
                comment = f"{comment} special {self.text(1)} requests"

            orders.append(
                (
                    key,
                    self.random.randrange(1, self.customers + 1),
                    self.random.choice(["O", "F", "P"]),
                    round(total, 2),
                    orderdate.isoformat(),
                    self.random.choice(PRIORITIES),
                    f"Clerk#{self.random.randrange(1, 1000):09}",
                    0,
                    comment,
                )
            )
            lineitem.extend(lines)

        return orders, lineitem


def write_table(directory: str, table: str, rows: t.Iterable[tuple]) -> int:
    columns = TABLE_COLUMNS[table]
    count = 0

    with gzip.open(os.path.join(directory, f"{table}.csv.gz"), "wt", newline="") as f:
        f.write("|".join(columns) + "\n")

        for row in rows:
            f.write("|".join("" if v is None else str(v) for v in row) + "\n")
            count += 1

    return count


def generate(out: str, scale: float, seed: int) -> dict[str, int]:
    os.makedirs(out, exist_ok=True)
    generator = Generator(scale, seed)
    counts = {
        "region": write_table(out, "region", generator.region()),
        "nation": write_table(out, "nation", generator.nation()),
        "part": write_table(out, "part", generator.part()),
        "supplier": write_table(out, "supplier", generator.supplier()),
        "partsupp": write_table(out, "partsupp", generator.partsupp()),
        "customer": write_table(out, "customer", generator.customer()),
    }
    orders, lineitem = generator.orders_and_lineitem()
    counts["orders"] = write_table(out, "orders", orders)
    counts["lineitem"] = write_table(out, "lineitem", lineitem)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scale",
        type=float,
        default=0.002,
        help="dbgen scale factor; the default keeps every table under ~10k rows",
    )
    parser.add_argument("--seed", type=int, default=0, help="seed, for reproducible output")
    parser.add_argument("--out", default="tpch-data", help="directory to write the CSVs to")
    parser.add_argument(
        "--run",
        action="store_true",
        help="after generating, run the 22 TPC-H queries through the engine",
    )
    args = parser.parse_args()

    for table, count in generate(args.out, args.scale, args.seed).items():
        print(f"{table:10} {count:>8} rows")

    print(f"\nwrote {args.out}")

    if args.run:
        from benchmarks.tpc_run import run

        run(args.out)


if __name__ == "__main__":
    main()
