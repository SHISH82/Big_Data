"""Среднегодовые температуры Воронежа, станция 34123, 1986–2025 годы."""

import argparse
import csv
from datetime import date
from pathlib import Path
import ssl
import sys
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup
import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE_URL = "https://www.pogodaiklimat.ru/history/34123.htm"
START_YEAR, END_YEAR = 1986, 2025
DATA_FILE = ROOT / "data" / "voronezh_annual.csv"


def parse_page(html):
    soup = BeautifulSoup(html, "html.parser")
    title = soup.find("h1")
    if title is None or "воронеж" not in title.get_text().lower():
        raise ValueError("Источник должен содержать таблицу температур Воронежа")
    left = soup.select_one(".chronicle-table-left-column table")
    right = soup.select_one(".chronicle-table table")
    if left is None or right is None:
        raise ValueError("Не найдены таблицы годов и температур")
    year_rows, temperature_rows = left.find_all("tr")[1:], right.find_all("tr")[1:]
    if len(year_rows) != len(temperature_rows):
        raise ValueError("Таблицы годов и температур не согласованы")
    result = []
    for year_row, temperature_row in zip(year_rows, temperature_rows):
        year = int(year_row.find("td").get_text(strip=True))
        if not START_YEAR <= year <= END_YEAR:
            continue
        cells = temperature_row.find_all("td")
        if len(cells) != 13:
            raise ValueError(f"Ожидались 12 месяцев и столбец «за год» для {year}")
        try:
            temp = float(cells[12].get_text(strip=True).replace(",", "."))
        except ValueError as error:
            raise ValueError(f"Нет среднегодовой температуры за {year}") from error
        result.append({"year": year, "temp": temp, "source_url": SOURCE_URL,
                       "retrieved_at": date.today().isoformat()})
    return validate_data(result)


def validate_data(records):
    converted = [{**row, "year": int(row["year"]), "temp": float(row["temp"])} for row in records]
    expected = list(range(START_YEAR, END_YEAR + 1))
    if [row["year"] for row in converted] != expected:
        raise ValueError("Нужны ровно 40 последовательных годов, 1986–2025, без повторов и пропусков")
    if any(not np.isfinite(row["temp"]) or abs(row["temp"]) >= 100 for row in converted):
        raise ValueError("Пропуск, маркер 999.9 или недопустимая годовая температура")
    if any(row["source_url"] != SOURCE_URL or not row["retrieved_at"] for row in converted):
        raise ValueError("Не указаны источник Воронежа или дата загрузки")
    return converted


def load_weather(refresh=False):
    if DATA_FILE.exists() and not refresh:
        with DATA_FILE.open(encoding="utf-8", newline="") as stream:
            return validate_data(list(csv.DictReader(stream)))
    context = ssl.create_default_context()
    if sys.platform == "darwin" and Path("/etc/ssl/cert.pem").exists():
        context.load_verify_locations("/etc/ssl/cert.pem")
    request = Request(SOURCE_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=30, context=context) as response:
        records = parse_page(response.read().decode("utf-8"))
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with DATA_FILE.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("year", "temp", "source_url", "retrieved_at"),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    return records


def main():
    parser = argparse.ArgumentParser(description="Температуры Воронежа за 1986–2025 годы")
    parser.add_argument("--refresh", action="store_true", help="Повторно скачать данные из источника")
    args = parser.parse_args()
    records = load_weather(refresh=args.refresh)
    print(f"Воронеж, станция 34123: {len(records)} годовых температур, {START_YEAR}–{END_YEAR}")
    print(f"Источник: {SOURCE_URL}\nФайл: {DATA_FILE}")


if __name__ == "__main__":
    main()
