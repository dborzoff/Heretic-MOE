# -*- coding: utf-8 -*-
r"""Оценщик перплексии для heretic: цена правки, измеренная по тексту.

Зачем он нужен вместо KL.

Штатный KLDivergence берёт сто безобидных промптов и сравнивает распределение
ПЕРВОГО токена ответа с исходной моделью. Это сто чисел. Перплексия на полном
тесте wikitext - двести тысяч токенов. Разница не в масштабе, а в том, что они
меряют: KL по первому токену видит, изменился ли выбор начала ответа, и почти
слеп к тому, что происходит дальше.

Наши замеры показали, насколько это расходится. Две точки одного фронта:

  сборка      KL (первый токен)   перплексия к оригиналу
  balanced         0.0021               +3.4%
  max              0.0126              +18.1%

По KL разница шестикратная, а по тексту - пятикратная при куда большей
абсолютной цене. Поиск расставлял точки на кривой, настоящей цены которой не
знал: он оптимизировал суррогат.

Раньше перплексию в цикл поиска было не вставить - на процессоре полный прогон
стоил четырнадцать минут. На карте те же данные считаются за секунды, и
суррогат больше не нужен.

Что считает этот оценщик. Средний отрицательный логарифм правдоподобия на
кусках текста, приведённый к относительному росту против исходной модели:

    value = perplexity / perplexity_baseline - 1

Ноль означает "текст модель предсказывает как раньше", 0.03 - три процента
хуже. Величина порядка единицы, как требует интерфейс Scorer.

Установка: положить рядом со штатными оценщиками и включить в настройках:

    [[scorers]]
    plugin = "heretic.scorers.perplexity.Perplexity"
    optimization = "minimize"
"""
import torch
from pydantic import BaseModel, Field

from heretic.config import DatasetSpecification
from heretic.plugin import Context
from heretic.scorer import Score, Scorer
from heretic.utils import print


class Settings(BaseModel):
    text: DatasetSpecification = Field(
        default=DatasetSpecification(
            dataset="wikitext",
            split="test",
            column="text",
        ),
        description="Текстовый набор, на котором меряется перплексия.",
    )
    window: int = Field(
        default=512,
        description="Длина куска в токенах. Та же, что у llama-perplexity, "
                    "чтобы числа были сопоставимы с замерами на гуфах.",
    )
    chunks: int = Field(
        default=24,
        description=(
            "Сколько кусков брать. Двадцать четыре - это 12 тысяч токенов и "
            "пара секунд на карте. Больше нужно для публикуемой цифры, а для "
            "сравнения испытаний между собой хватает: текст один и тот же, и "
            "разница считается на одних и тех же кусках."
        ),
    )


class Perplexity(Scorer):
    """
    Perplexity on a fixed text corpus, relative to the baseline model.
    Measures how much the edit degraded language modelling.
    Lower is better (less damage).
    """

    settings: Settings

    @property
    def reproducible(self) -> bool:
        return True

    @property
    def score_name(self) -> str:
        return "Perplexity increase"

    # Context намеренно не отдаёт саму модель наружу, а прогон по произвольному
    # тексту через его методы не выразить: get_logits работает с промптами и
    # возвращает только последний токен. Берём модель напрямую и держим это
    # в одном месте, чтобы при обновлении heretic чинить пришлось только здесь.
    @staticmethod
    def _model_and_tokenizer(ctx: Context):
        m = ctx._model            # noqa: SLF001
        return m.model, m.tokenizer

    def _windows(self, ctx: Context):
        """Нарезать текст на куски по window токенов - один раз за прогон."""
        from datasets import load_dataset

        spec = self.settings.text
        ds = load_dataset(spec.dataset, split=spec.split) \
            if "/" in spec.dataset or spec.dataset != "wikitext" \
            else load_dataset("wikitext", "wikitext-2-raw-v1", split=spec.split)
        text = "\n\n".join(t for t in ds[spec.column] if t.strip())

        _, tok = self._model_and_tokenizer(ctx)
        ids = tok(text, return_tensors="pt").input_ids[0]
        w = self.settings.window
        n = min(self.settings.chunks, len(ids) // w)
        return [ids[i * w:(i + 1) * w] for i in range(n)]

    @torch.no_grad()
    def _perplexity(self, ctx: Context) -> float:
        model, _ = self._model_and_tokenizer(ctx)
        device = next(model.parameters()).device
        total, count = 0.0, 0
        for w in self._windows_cached:
            ids = w.unsqueeze(0).to(device)
            # Метки те же, что вход: модель сама сдвигает их на один токен.
            out = model(ids, labels=ids)
            # loss - средний NLL по куску; складываем взвешенно по токенам,
            # чтобы неполный последний кусок не перевесил остальные.
            total += float(out.loss) * (ids.shape[1] - 1)
            count += ids.shape[1] - 1
        return float(torch.exp(torch.tensor(total / max(count, 1))))

    def init(self, ctx: Context) -> None:
        print()
        print(f"Loading Perplexity text from [bold]{self.settings.text.dataset}[/]...")
        self._windows_cached = self._windows(ctx)
        print(f"* [bold]{len(self._windows_cached)}[/] windows of "
              f"[bold]{self.settings.window}[/] tokens")
        print("* Measuring baseline perplexity...")
        self._baseline = self._perplexity(ctx)
        print(f"* Baseline perplexity: [bold]{self._baseline:.4f}[/]")

    def get_score(self, ctx: Context) -> Score:
        ppl = self._perplexity(ctx)
        rel = ppl / self._baseline - 1.0
        return Score(
            value=rel,
            rich_display=f"[bold]{ppl:.4f}[/] ([bold]{rel * 100:+.2f}%[/] vs baseline)",
            md_display=f"{ppl:.4f} ({rel * 100:+.2f}%)",
        )
