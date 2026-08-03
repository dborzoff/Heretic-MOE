# -*- coding: utf-8 -*-
r"""Править экспертов не поровну, а по их отклику на направление отказа.

Что не так сейчас. Все 256 экспертов слоя получают один и тот же вес правки.
Но измерение на настоящих весах Qwen3.6 показало, что вдоль осмысленного
направления эксперты откликаются по-разному: сильнейший впятеро сильнее
слабейшего, верхняя четверть несёт около сорока процентов отклика вместо
равномерных двадцати пяти.

Значит половина экспертов почти не участвует в направлении, но правится в
полную силу - и платит за это отклонением от исходной модели. Отсюда и резкий
рост KL, который мы видели при попытке добить отказы ниже 0.10.

Что делает правка. Масштабирует вес по каждому эксперту:

    p_e   = ||v^T W_e||                норма проекции выхода на направление
    s_e   = (p_e / max_e p_e) ** gamma доля от сильнейшего, в степени gamma
    lambda_e = lambda * s_e

Показатель gamma ищется наравне с остальными:

    gamma = 0   все эксперты поровну - нынешнее поведение
    gamma = 1   строго пропорционально отклику
    gamma > 1   ещё резче, только самые причастные

Дополнительного прогона модели не нужно: v и W_e уже под рукой в том же месте,
где считается правка. Стоимость - одна свёртка на слой.

Наложение:
    python per-expert-scaling.patch.py <путь к model.py> <путь к main.py>
"""
import io
import sys

OLD_MODEL = '''                if keep_norm:'''

NEW_MODEL = '''                # Отклик каждого эксперта на направление: чем он меньше, тем
                # меньше смысла трогать эксперта - в отказы он почти не вносит,
                # а искажение от правки вносит наравне со всеми.
                if gamma > 0.0:
                    p = torch.einsum("h,ehi->ei", v, W).norm(dim=1)
                    scale = (p / p.max().clamp(min=1e-12)).pow(gamma).view(-1, 1, 1)
                else:
                    scale = 1.0

                if keep_norm:'''

# в обеих ветвях домножаем вес на посчитанный масштаб
OLD_A = '''                    W = W - weight * v.view(1, -1, 1) * proj.unsqueeze(1)
                    W = W / W.norm(dim=2, keepdim=True).clamp(min=1e-12)
                    W = W * m'''
NEW_A = '''                    W = W - (weight * scale) * v.view(1, -1, 1) * proj.unsqueeze(1)
                    W = W / W.norm(dim=2, keepdim=True).clamp(min=1e-12)
                    W = W * m'''

OLD_B = '''                    proj = torch.einsum("h,ehi->ei", v, W)
                    W = W - weight * v.view(1, -1, 1) * proj.unsqueeze(1)

                fused.data[lo:hi].copy_'''
NEW_B = '''                    proj = torch.einsum("h,ehi->ei", v, W)
                    W = W - (weight * scale) * v.view(1, -1, 1) * proj.unsqueeze(1)

                fused.data[lo:hi].copy_'''

# gamma достаётся из параметров испытания
OLD_G = '''        params = parameters["mlp.down_proj"]'''
NEW_G = '''        params = parameters["mlp.down_proj"]
        gamma = getattr(params, "expert_gamma", 0.0) or 0.0'''

# dataclass не примет лишний аргумент - добавляем поле со значением по умолчанию,
# тогда все существующие вызовы конструктора продолжат работать как были.
OLD_DC = '''    min_weight: float
    min_weight_distance: float'''
NEW_DC = '''    min_weight: float
    min_weight_distance: float
    # Насколько резко ослаблять правку у экспертов со слабым откликом на
    # направление. 0 - всем поровну, как было до этой правки.
    expert_gamma: float = 0.0'''

OLD_CFG = '''            min_weight_distance = trial.suggest_float('''
NEW_CFG = '''            # Для сросшихся экспертов ищем ещё и резкость ослабления.
            expert_gamma = (
                trial.suggest_float("expert_gamma", 0.0, 2.0)
                if component == "mlp.down_proj" else 0.0
            )
            min_weight_distance = trial.suggest_float('''

OLD_AP = '''                min_weight=(min_weight * max_weight),
                min_weight_distance=min_weight_distance,
            )'''
NEW_AP = '''                min_weight=(min_weight * max_weight),
                min_weight_distance=min_weight_distance,
                expert_gamma=expert_gamma,
            )'''


def apply(path, pairs):
    src = io.open(path, encoding="utf-8").read()
    for old, new, label in pairs:
        marker = new.strip().splitlines()[0]
        if marker in src and old not in src:
            print(f"  {label}: уже наложено")
            continue
        if old not in src:
            print(f"  {label}: МЕСТО НЕ НАЙДЕНО")
            return False
        src = src.replace(old, new, 1)
        print(f"  {label}: наложено")
    io.open(path, "w", encoding="utf-8", newline="\n").write(src)
    return True


if __name__ == "__main__":
    model_py, main_py = sys.argv[1], sys.argv[2]
    ok = apply(model_py, [
        (OLD_DC, NEW_DC, "поле в структуре параметров"),
        (OLD_G, NEW_G, "чтение gamma"),
        (OLD_MODEL, NEW_MODEL, "расчёт отклика по экспертам"),
        (OLD_A, NEW_A, "масштаб в ветви с сохранением нормы"),
        (OLD_B, NEW_B, "масштаб в обычной ветви"),
    ])
    ok = ok and apply(main_py, [
        (OLD_CFG, NEW_CFG, "поиск по gamma"),
        (OLD_AP, NEW_AP, "передача gamma в конструктор"),
    ])
    print("  ГОТОВО" if ok else "  НЕ НАЛОЖЕНО ЦЕЛИКОМ")
    sys.exit(0 if ok else 1)
