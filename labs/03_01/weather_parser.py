"""Температуры Воронежа за 2024–2025 годы: загрузка и проверка CSV."""

import argparse
import calendar
from datetime import date
from html import unescape
from pathlib import Path
import re
import ssl
import sys
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
STATION = "34123"
YEARS = (2024, 2025)


def weather_url(year, month):
    return f"https://www.pogodaiklimat.ru/monitor.php?id={STATION}&month={month}&year={year}"


def parse_month(html, year, month):
    """Извлечь столбец «Средняя» из первой таблицы климатического монитора."""
    table = re.search(r"<table\b[^>]*>(.*?)</table>", html, re.S | re.I)
    if table is None:
        raise ValueError(f"Не найдена таблица за {year}-{month:02}")
    records = []
    for row in re.findall(r"<tr\b[^>]*>(.*?)</tr>", table[1], re.S | re.I):
        cells = [unescape(re.sub(r"<[^>]*>", "", cell)).strip()
                 for cell in re.findall(r"<td\b[^>]*>(.*?)</td>", row, re.S | re.I)]
        if len(cells) < 6 or not cells[0].isdigit():
            continue
        day = int(cells[0])
        # Сайт рисует 31 строку даже для февраля; несуществующие даты пропускаем.
        if not 1 <= day <= calendar.monthrange(year, month)[1]:
            continue
        try:
            temperature = float(cells[2].replace(",", "."))
        except ValueError as error:
            raise ValueError(f"Нет средней температуры за {year}-{month:02}-{day:02}") from error
        records.append({
            "date": date(year, month, day).isoformat(), "temperature": temperature,
            "source_url": weather_url(year, month), "retrieved_at": date.today().isoformat(),
        })
    expected = calendar.monthrange(year, month)[1]
    if len(records) != expected or len({row["date"] for row in records}) != expected:
        raise ValueError(f"Неполная или повторяющаяся таблица за {year}-{month:02}")
    return records


def validate_weather(data):
    data = data.copy()
    required = {"date", "temperature", "source_url", "retrieved_at"}
    if not required.issubset(data.columns):
        raise ValueError(f"В CSV нужны столбцы {sorted(required)}")
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    expected = pd.date_range(f"{YEARS[0]}-01-01", f"{YEARS[-1]}-12-31", freq="D")
    if (not pd.DatetimeIndex(data["date"]).equals(expected)
            or not np.isfinite(data["temperature"]).all()):
        raise ValueError("Нужен непрерывный ряд из 731 температуры за 2024–2025 годы")
    sources = [weather_url(d.year, d.month) for d in data["date"]]
    if data["source_url"].tolist() != sources:
        raise ValueError("CSV должен содержать данные станции Воронеж 34123")
    return data


def load_weather(refresh=False):
    filename = ROOT / "data" / "temperature.csv"
    if filename.exists() and not refresh:
        return validate_weather(pd.read_csv(filename))
    context = ssl.create_default_context()
    # Дополняем доверенные CA системными сертификатами macOS.
    if sys.platform == "darwin" and Path("/etc/ssl/cert.pem").exists():
        context.load_verify_locations("/etc/ssl/cert.pem")
    records = []
    for year in YEARS:
        for month in range(1, 13):
            print(f"Загрузка температур Воронежа: {year}-{month:02}", flush=True)
            request = Request(weather_url(year, month), headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=30, context=context) as response:
                html = response.read().decode("utf-8")
            records.extend(parse_month(html, year, month))
    data = validate_weather(pd.DataFrame(records))
    filename.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(filename, index=False)
    return data


def main():
    parser = argparse.ArgumentParser(description="Температуры Воронежа, станция 34123, 2024–2025")
    parser.add_argument("--refresh", action="store_true", help="Обновить сохранённые данные из интернета")
    args = parser.parse_args()
    data = load_weather(refresh=args.refresh)
    print(f"Воронеж: {len(data)} суток, с {data.date.min():%d.%m.%Y} по {data.date.max():%d.%m.%Y}")
    print(f"Данные: {ROOT / 'data' / 'temperature.csv'}")


if __name__ == "__main__":
    main()
