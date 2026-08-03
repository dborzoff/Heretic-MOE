# -*- coding: utf-8 -*-
r"""Сохранить переданное в командной строке при возобновлении исследования.

Что не так. При `--checkpoint-action continue` heretic целиком заменяет
настройки теми, что лежат в журнале:

    settings = Settings.model_validate_json(existing_study.user_attrs["settings"])

Часть полей туда вообще не попадает - `save_directory` и `upload_repo_id`
объявлены с `exclude=True`. После подмены они None, и heretic начинает
спрашивать путь у пользователя через questionary. В сессии без ввода это
тихо заканчивается: ни ошибки, ни трассировки, процесс просто исчезает.

Правка запоминает настройки до подмены и возвращает поверх неё всё, что было
задано явно.

Наложение:
    python cli-survives-resume.patch.py <путь к main.py>
"""
import io
import sys

OLD = '''        if action == "continue":
            settings = Settings.model_validate_json(
                existing_study.user_attrs["settings"]
            )'''

NEW = '''        if action == "continue":
            _cli = settings
            settings = Settings.model_validate_json(
                existing_study.user_attrs["settings"]
            )
            # Podmena nastroek sohranyonnymi steraet peredannoe v komandnoy
            # stroke. Chast poley (save_directory, upload_repo_id) pomechena
            # exclude=True i v zhurnal ne popadaet vovse - posle podmeny oni
            # None, i heretic nachinaet sprashivat put u polzovatelya.
            for _f in ("save_directory", "model_action", "upload_repo_id",
                       "trial_index", "batch_size", "export_strategy"):
                _v = getattr(_cli, _f, None)
                if _v is not None:
                    setattr(settings, _f, _v)
                    print(f"iz komandnoy stroki: {_f}={_v}")'''


def main():
    path = sys.argv[1]
    src = io.open(path, encoding="utf-8").read()
    if "_cli = settings" in src:
        print("  уже наложено")
        return 0
    if OLD not in src:
        print("  МЕСТО НЕ НАЙДЕНО")
        return 1
    io.open(path, "w", encoding="utf-8", newline="\n").write(src.replace(OLD, NEW))
    print("  наложено")
    return 0


if __name__ == "__main__":
    sys.exit(main())
