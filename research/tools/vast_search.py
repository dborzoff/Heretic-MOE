# -*- coding: utf-8 -*-
"""Подбор машины на vast.ai по полной стоимости прогона, а не по ставке.

Стоимость складывается из четырёх кусков, и три из них зависят от машины:

  1. СТАРТ    скачать модель (75 ГБ) - это и трафик, и время: канал 300 Мбит/с
              тянет её 33 минуты, канал 3 Гбит/с - три. Аренда идёт всё это время.
              Плюс развернуть окружение: собрать llama.cpp с CUDA, поставить
              зависимости. Это фиксированные ~25 минут независимо от машины.
  2. РАБОТА   сам прогон. Часы задаются сверху, число карт может их сократить.
  3. ФИНИШ    залить результат на HF (~500 ГБ на сборку) - снова трафик и время.
  4. ДИСК     хранилище за все эти часы.

Главное, чего не хватало в прежней версии: время передачи тоже оплачивается по
часовой ставке. Дешёвый трафик на медленном канале обходится дороже дорогого
трафика на быстром.

Сегодняшний счёт, для калибровки: 4.58 часа = $4.56 аренды и $6.11 трафика
(937 ГБ внутрь, 939 наружу).
"""
import json
import subprocess

KEY = open(r"F:\AI\vaskAI_api_key").read().strip()

DOWN_GB = 75        # модель на старте
UP_GB = 500         # выложить одну сборку со всеми форматами
SETUP_H = 0.42      # сборка llama.cpp с CUDA и зависимости, ~25 минут
WORK_H = 4.0        # сам прогон поиска
DISK_GB = 500
# У Max-Q около 300 Вт, у полной RTX PRO 6000 - 550-600. Между ними попадаются
# 400-500: это хост придушил карту, и скорость падает вместе с питанием.
MIN_WATT = 500


def api(path, body=None, ver="v0"):
    cmd = ["curl", "-sL", "-H", f"Authorization: Bearer {KEY}",
           "-H", "Content-Type: application/json"]
    if body is not None:
        cmd += ["-X", "POST", "-d", json.dumps(body)]
    cmd.append(f"https://console.vast.ai/api/{ver}/{path}")
    try:
        return json.loads(subprocess.run(cmd, capture_output=True).stdout or "{}")
    except Exception:
        return {}


def hours(gb, mbps):
    """Часы на передачу. Мбит/с -> ГБ/ч: делим на 8 и умножаем на 3600/1000."""
    return (gb * 8 * 1000) / (mbps * 3600) if mbps else 99.0


def breakdown(dph, cd, cu, storage_month, down_mbps, up_mbps,
              down_gb=DOWN_GB, up_gb=UP_GB, work_h=WORK_H):
    """Разложение стоимости прогона по кускам."""
    h_down = hours(down_gb, down_mbps)
    h_up = hours(up_gb, up_mbps)
    h_all = h_down + SETUP_H + work_h + h_up
    rent = h_all * dph
    traffic = down_gb * cd + up_gb * cu
    disk = (storage_month or 0) * DISK_GB / 730 * h_all
    return dict(итого=rent + traffic + disk, аренда=rent, трафик=traffic,
                диск=disk, часы=h_all, ч_вниз=h_down, ч_вверх=h_up)


def main():
    cur = None
    for i in api("instances", ver="v1").get("instances", []):
        cur = i
        break

    if cur:
        # Модель на ней уже лежит: ни трафика вниз, ни времени на закачку.
        b = breakdown(cur.get("dph_total") or 0, 0, cur.get("inet_up_cost") or 0,
                      0, 1e9, cur.get("inet_up") or 1000, down_gb=0)
        print(f"  ОСТАТЬСЯ: {cur.get('gpu_name')}, {cur.get('geolocation')}, "
              f"${cur.get('dph_total'):.4f}/ч")
        print(f"    модель уже здесь, окружение развёрнуто")
        print(f"    работа {WORK_H:.1f} ч + заливка {b['ч_вверх']:.1f} ч "
              f"= {b['часы']:.1f} ч аренды ${b['аренда']:.2f} "
              f"+ трафик ${b['трафик']:.2f} = ${b['итого']:.2f}\n")
        t_stay = b["итого"]
    else:
        t_stay = None
        print("  текущей машины не вижу\n")

    query = {
        "verified": {"eq": True}, "rentable": {"eq": True}, "rented": {"eq": False},
        "num_gpus": {"gte": 1, "lte": 6},
        "gpu_ram": {"gte": 90000},
        "disk_space": {"gte": DISK_GB},
        "inet_down": {"gte": 100}, "inet_up": {"gte": 100},
        "reliability2": {"gte": 0.98},
        "gpu_max_power": {"gte": MIN_WATT},
        "type": "ondemand", "order": [["dph_total", "asc"]], "limit": 300,
    }
    offers = api("bundles", query).get("offers", [])
    print(f"  предложений: {len(offers)}\n")
    if not offers:
        print("  ничего не подошло")
        return

    rows = []
    for o in offers:
        b = breakdown(o.get("dph_total") or 0, o.get("inet_down_cost") or 0,
                      o.get("inet_up_cost") or 0, o.get("storage_cost"),
                      o.get("inet_down") or 0, o.get("inet_up") or 0)
        rows.append((b, o))
    rows.sort(key=lambda r: r[0]["итого"])

    perfs = sorted((o.get("dlperf") or 0) / max(o.get("num_gpus") or 1, 1)
                   for _, o in rows)
    med = perfs[len(perfs) // 2] if perfs else 0

    print(f"  {'итого':>7} {'аренда':>7} {'трафик':>7} {'диск':>5}  "
          f"{'часы':>5} {'вниз':>5} {'вверх':>5}  {'$/ч':>6} {'карт':>4} "
          f"{'ватт':>5} {'скор':>5}  где / карта")
    for b, o in rows[:14]:
        name = o.get("gpu_name") or ""
        watt = o.get("gpu_max_power") or 0
        perf = (o.get("dlperf") or 0) / max(o.get("num_gpus") or 1, 1)
        flag = ""
        if "max-q" in name.lower() or (watt and watt < MIN_WATT):
            flag = "  <- придушена по питанию"
        elif med and perf < med * 0.5:
            flag = f"  <- скорость {perf:.0f} против {med:.0f}, брать нельзя"
        elif b["трафик"] == 0:
            flag = "  <- трафик даром"
        print(f"  ${b['итого']:6.2f} ${b['аренда']:6.2f} ${b['трафик']:6.2f} "
              f"${b['диск']:4.2f}  {b['часы']:5.1f} {b['ч_вниз']:5.2f} "
              f"{b['ч_вверх']:5.2f}  ${o.get('dph_total'):5.3f} "
              f"{o.get('num_gpus'):>4} {watt:>5.0f} {perf:>5.0f}  "
              f"{(o.get('geolocation') or '?')[:12]} / {name[:20]}{flag}")

    if t_stay is not None:
        b, o = rows[0]
        print(f"\n  ОСТАТЬСЯ:  ${t_stay:.2f}")
        print(f"  ПЕРЕЕХАТЬ: ${b['итого']:.2f}  ({o.get('num_gpus')}x "
              f"{o.get('gpu_name')}, {o.get('geolocation')}) - из них "
              f"${b['аренда'] - WORK_H * (o.get('dph_total') or 0):.2f} "
              f"уходит на закачку и настройку")
        d = t_stay - b["итого"]
        print(f"  {'выгода переезда' if d > 0 else 'переезд дороже'}: ${abs(d):.2f}")


if __name__ == "__main__":
    main()
