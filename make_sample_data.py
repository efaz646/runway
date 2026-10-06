"""Build Dave's sample data for Bradford Auto Care (a made-up garage in Bradford).

Writes two files with known answers, so every test is a simple pass or fail:
  * bank_statement.csv: 1 Apr to 4 Oct 2026, deliberately messy (mixed date formats,
    mixed amount formats, newest first, stray spaces and blank rows).
  * invoices.csv: trade customer invoices with paid status.

Known answers, as of Monday 5 October 2026 (config.AS_OF_DATE):
  * cash runs short in week 5 (week commencing 2 Nov), when the quarterly VAT
    payment and the monthly parts account both fall due;
  * exactly 3 invoices are overdue (12, 31 and 49 days);
  * exactly 1 supplier bill spike: the September electricity bill.

Run:  python make_sample_data.py
"""

import csv
import random
from datetime import date, timedelta

import config

START = date(2026, 4, 1)
END = config.AS_OF_DATE - timedelta(days=1)  # Sunday 4 Oct 2026
CLOSING_BALANCE = 3600.00  # balance on 4 Oct; the opening balance is worked back from it
SEED = 7

# Regular monthly payments: (day of month, description, amounts by month Apr..Sep[, Oct])
ELECTRICITY = [412.60, 398.20, 386.40, 379.95, 391.30, 1236.80]  # September is the spike
PARTS_ACCOUNT = [2382.15, 2511.40, 2447.80, 2463.25, 2391.60, 2452.90]
MONTHLY = [
    (1, "DD CANAL ROAD ESTATES RENT", [1850.00] * 7),  # Apr..Oct (1 Oct is in the statement)
    (5, "DD PENNINE MOTOR FACTORS ACC 40213", PARTS_ACCOUNT),
    (10, "DD WHITE ROSE INSURANCE POL 77-310", [162.40] * 6),
    (15, "DD NORTHERN GRID ENERGY REF 88213", ELECTRICITY),
    (20, "DD AIRE VALLEY FINANCE LIFT AGR", [385.00] * 6),
    (22, "DD DALES TELECOM BROADBAND", [68.99] * 6),
    (12, "OIL RECYCLING YORKS LTD", [60.00] * 6),
    (28, "BANK CHARGES", [18.50] * 6),
]
VAT_PAYMENTS = [(date(2026, 5, 7), 4418.36), (date(2026, 8, 7), 4781.64)]
WAGES = 2050.00
WAGES_WITH_OVERTIME = 2186.40
OVERTIME_FRIDAYS = {date(2026, 5, 22), date(2026, 7, 3), date(2026, 8, 28), date(2026, 9, 25)}

# Trade customer invoices: (number, customer, contact, email, amount, due, paid_on or None)
INVOICES = [
    ("INV-1003", "Idle Cabs Ltd", "Sandra", "accounts@idlecabs.example", 640.00, date(2026, 4, 10), date(2026, 4, 9)),
    ("INV-1006", "Thornton Builders", "Gary", "office@thorntonbuilders.example", 1180.00, date(2026, 4, 24), date(2026, 4, 27)),
    ("INV-1009", "Shipley Van Hire", "Priya", "accounts@shipleyvanhire.example", 2210.00, date(2026, 5, 8), date(2026, 5, 8)),
    ("INV-1012", "Saltaire Catering Co", "morty", "tom@saltairecatering.example", 455.50, date(2026, 5, 22), date(2026, 5, 21)),
    ("INV-1015", "Wharfe Valley Taxis", "Mick", "mick@wharfevalleytaxis.example", 1540.00, date(2026, 6, 5), date(2026, 6, 9)),
    ("INV-1018", "Calder Plumbing & Heating", "Joanne", "accounts@calderplumbing.example", 980.00, date(2026, 6, 19), date(2026, 6, 19)),
    ("INV-1021", "Airedale Couriers", "Imran", "imran@airedalecouriers.example", 1325.00, date(2026, 7, 3), date(2026, 7, 6)),
    ("INV-1024", "Idle Cabs Ltd", "Sandra", "accounts@idlecabs.example", 715.00, date(2026, 7, 17), date(2026, 7, 16)),
    ("INV-1027", "Thornton Builders", "Gary", "office@thorntonbuilders.example", 860.00, date(2026, 7, 31), date(2026, 8, 4)),
    ("INV-1029", "Shipley Van Hire", "Priya", "accounts@shipleyvanhire.example", 2475.00, date(2026, 8, 17), None),  # overdue 49 days
    ("INV-1033", "Bingley Florists", "Helen", "helen@bingleyflorists.example", 298.00, date(2026, 8, 28), date(2026, 8, 27)),
    ("INV-1035", "Saltaire Catering Co", "morty", "tom@saltairecatering.example", 512.00, date(2026, 9, 11), date(2026, 9, 14)),
    ("INV-1037", "Calder Plumbing & Heating", "Joanne", "accounts@calderplumbing.example", 1240.50, date(2026, 9, 4), None),  # overdue 31 days
    ("INV-1039", "Idle Cabs Ltd", "Sandra", "accounts@idlecabs.example", 690.00, date(2026, 9, 25), date(2026, 9, 24)),
    ("INV-1042", "Wharfe Valley Taxis", "Mick", "mick@wharfevalleytaxis.example", 1860.00, date(2026, 9, 23), None),  # overdue 12 days
    ("INV-1047", "Bingley Florists", "Helen", "helen@bingleyflorists.example", 385.00, date(2026, 10, 16), None),  # due week 2
    ("INV-1049", "Airedale Couriers", "Imran", "imran@airedalecouriers.example", 1120.00, date(2026, 10, 28), None),  # due week 4
]


def days(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def build_transactions(rng: random.Random) -> list[tuple[date, str, float]]:
    tx: list[tuple[date, str, float]] = []
    span = (END - START).days

    for d in days(START, END):
        # Card takings Monday to Saturday, drifting down from about £3,350 to £3,150 a week.
        if d.weekday() < 6:
            weekly_level = 3350 - 200 * (d - START).days / span
            day_shape = [0.90, 0.95, 1.00, 1.00, 1.15, 1.00][d.weekday()]
            amount = round(weekly_level / 6 * day_shape * rng.uniform(0.9, 1.1), 2)
            tx.append((d, f"TILLPOINT SETTLEMENT {d:%d%m}", amount))
        # Wages every Friday.
        if d.weekday() == 4:
            pay = WAGES_WITH_OVERTIME if d in OVERTIME_FRIDAYS else WAGES
            tx.append((d, f"BACS WAGES RUN {d:%d%m}", -pay))
        # Fuel for the courtesy car every Monday, and tools or consumables most weeks.
        if d.weekday() == 0:
            tx.append((d, "FUEL STN THORNTON RD", -round(rng.uniform(45, 75), 2)))
        if d.weekday() == 2 and rng.random() < 0.6:
            tx.append((d, "TRADE TOOLS DIRECT", -round(rng.uniform(30, 140), 2)))

    for day_of_month, description, amounts in MONTHLY:
        for i, amount in enumerate(amounts):
            d = date(2026, 4 + i, day_of_month)
            if START <= d <= END:
                tx.append((d, description, -amount))

    for d, amount in VAT_PAYMENTS:
        tx.append((d, "HMRC VAT 123456789", -amount))

    for number, customer, *_rest, amount, _due, paid_on in INVOICES:
        if paid_on is not None:
            tx.append((paid_on, f"FPI {customer.upper()} {number}", amount))

    return tx


# ------------------------------------------------------------- messy formatting
DATE_FORMATS = ["%d/%m/%Y", "%d/%m/%Y", "%Y-%m-%d", "%d %b %Y", "%d-%m-%y"]


def messy_date(d: date, rng: random.Random) -> str:
    return d.strftime(rng.choice(DATE_FORMATS))


def messy_amount(amount: float, rng: random.Random) -> str:
    money = f"{abs(amount):,.2f}"
    if amount >= 0:
        return rng.choice([f"£{money}", money.replace(",", ""), f"+£{money}"])
    return rng.choice([f"-£{money}", f"-{money.replace(',', '')}", f"{money} DR", f"({money})"])


def messy_description(text: str, rng: random.Random) -> str:
    if rng.random() < 0.3:
        text = text.replace(" ", "  ", 1)
    if rng.random() < 0.2:
        text = f" {text} "
    return text


def main() -> None:
    rng = random.Random(SEED)
    tx = build_transactions(rng)
    opening = round(CLOSING_BALANCE - sum(a for _, _, a in tx), 2)
    tx.append((START, "BALANCE BROUGHT FORWARD", opening))
    tx.sort(key=lambda t: (t[0], t[1] != "BALANCE BROUGHT FORWARD"))

    # Lowest balance in the history, as a sanity check on the made-up numbers.
    running, lowest = 0.0, None
    for d, _, a in tx:
        running += a
        lowest = running if lowest is None else min(lowest, running)

    with open(config.BANK_STATEMENT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([" Date", "Description ", " Amount "])
        for i, (d, description, amount) in enumerate(reversed(tx)):  # newest first, like a bank export
            if i in (17, 64, 141):
                writer.writerow([])
            writer.writerow([messy_date(d, rng), messy_description(description, rng), messy_amount(amount, rng)])

    with open(config.INVOICES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["customer", "contact_name", "email", "invoice_number", "amount", "due_date", "paid"])
        for number, customer, contact, email, amount, due, paid_on in INVOICES:
            writer.writerow([customer, contact, email, number, f"{amount:.2f}", due.isoformat(), "Yes" if paid_on else "No"])

    print(f"Wrote {config.BANK_STATEMENT_CSV.name}: {len(tx)} transactions, "
          f"opening £{opening:,.2f}, closing £{CLOSING_BALANCE:,.2f}, lowest £{lowest:,.2f}")
    print(f"Wrote {config.INVOICES_CSV.name}: {len(INVOICES)} invoices")


if __name__ == "__main__":
    main()
