# -*- coding: utf-8 -*-
"""Прерываемые машины на vast под варку 35B, с полной стоимостью.

У vast прерываемые называются bid: платишь свою ставку, и пока её никто не
перебил - работаешь. Перебили - инстанс останавливают.

Считаем полную стоимость как обычно: ставка плюс трафик плюс диск. Урок дня:
у норвежской машины трафик по $0.04/ГБ съел половину счёта.
"""
import json
import subprocess

KEY = open(r"F:\AI\vaskAI_api_key").read().strip()
IN_GB, OUT_GB, HOURS = 72, 72, 2.5


def search(kind):
    q = {
        "verified": {"eq": True},
        "rentable": {"eq": True},
        "num_gpus": {"gte": 2, "lte": 8},
        "gpu_ram": {"gte": 90000},
        "disk_space": {"gte": 180},
        "inet_down": {"gte": 150},
        "reliability2": {"gte": 0.97},
        "type": kind,
        "limit": 100,
    }
    cmd = ["curl", "-sL", "-H", f"Authorization: Bearer {KEY}",
           "-H", "Content-Type: application/json", "-X", "POST",
           "-d", json.dumps(q), "https://console.vast.ai/api/v0/bundles"]
    return json.loads(subprocess.run(cmd, capture_output=True).stdout).get("offers", [])


for kind, label in (("bid", "ПРЕРЫВАЕМЫЕ (bid)"), ("ondemand", "обычные")):
    offers = search(kind)
    rows = []
    for o in offers:
        # у прерываемых ориентир - min_bid, это нижняя граница ставки
        dph = o.get("min_bid") if kind == "bid" else o.get("dph_total")
        dph = dph or o.get("dph_total") or 0
        n = max(o.get("num_gpus") or 1, 1)
        traffic = IN_GB * (o.get("inet_down_cost") or 0) + OUT_GB * (o.get("inet_up_cost") or 0)
        disk = (o.get("storage_cost") or 0) * 200 / 730 * HOURS
        # с большим числом карт поиск идёт быстрее: время делим на число работников
        hours = HOURS * 2 / n
        rows.append((dph * hours + traffic + disk, dph, traffic, disk, hours, o))

    print(f"\n═══ {label}: найдено {len(offers)}")
    if not rows:
        continue
    print(f"{'итого':>7} {'ставка':>7} {'трафик':>7} {'диск':>6} {'часов':>6} "
          f"{'карт':>4} {'надёж':>6}  где")
    for total, dph, traffic, disk, hours, o in sorted(rows, key=lambda r: r[0])[:8]:
        mark = "  <- трафик даром" if traffic == 0 else ""
        print(f"${total:6.2f} ${dph:6.3f} ${traffic:6.2f} ${disk:5.2f} {hours:6.2f} "
              f"{o.get('num_gpus'):>4} {(o.get('reliability2') or 0)*100:>5.1f}%  "
              f"{o.get('geolocation') or '?'}{mark}")
