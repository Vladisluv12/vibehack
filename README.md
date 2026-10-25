# Обучение распознавания жестов

Все команды ниже выполняются из корня проекта в WSL:

```bash
cd ~/study/projects/vibehack
source .venv/bin/activate
pip install -r requirements-training.txt
```

PyTorch с CUDA устанавливается отдельно. Уже установленный torch не заменяется
через requirements-training.txt. Обучение работает с готовыми .npy, камера не нужна.

Если pip в WSL отвечает сетевыми таймаутами, в этой рабочей копии подготовлены
официальные wheels scikit-learn и его недостающих зависимостей (SHA-256 проверены
по PyPI). Их можно установить без сети в активное окружение:

```bash
pip install --no-index --find-links runs/verification-wheels scikit-learn
```

Эта папка локальная и игнорируется Git. Проверки выполнялись с изолированными
зависимостями; окружение .venv не изменено.

## Расширение данных

В features.py словарь WORDS расширен с 24 до 72 классов:
71 жест и no_event. Метки проверены по локальному annotations.csv.
Для 48 новых жестов доступны ещё 960 видео: 720 train и 240 test.
После извлечения всего будет 1365 train и 455 test клипов.

В этой версии Slovo для каждого выбранного жеста есть 15 train и 5 test примеров.
extract_slovo.py уже выбирает все реальные примеры для каждого слова.
Увеличить число независимых примеров текущих жестов можно только добавлением
собственных записей/другого источника данных. Аугментации создаются случайно
во время обучения; они не записываются отдельными файлами и не попадают в валидацию.

Сначала проверить план, затем извлечь недостающие признаки:

```bash
python src/translator/extract_slovo.py --dry-run
python src/translator/extract_slovo.py --workers 4
```

Готовые файлы сохраняются и пропускаются. Можно выбрать подмножество слов:

```bash
python src/translator/extract_slovo.py --words privet poka voda dom --workers 4
```

Обучение берёт все доступные классы из WORDS, для которых есть готовые признаки
в train. Пока новые признаки не извлечены, оно выведет предупреждение о пропущенных
словах и будет работать с текущими классами. Для изменения самого набора слов
отредактируй WORDS в src/translator/features.py.

## Архитектуры и один эксперимент

- gru: исходная BiGRU со средним по времени; база для сравнения.
- attn_gru: два residual Conv1D блока, BiGRU, attention/mean/max pooling и MLP.
- transformer: позиционное кодирование, Transformer Encoder и attention/mean pooling.

```bash
python src/translator/train.py --arch attn_gru --hidden 192 --layers 2 \
  --dropout 0.35 --epochs 100 --patience 15 --device cuda
```

Флаги: --lr, --weight-decay, --batch-size, --seq-len, --heads (для Transformer),
--label-smoothing, --balance none|sqrt|sampler, --loss ce|focal,
--focal-gamma, --mixup (например, 0.2), --scheduler cosine|onecycle.
Флаг --amp включает mixed precision на CUDA. --num-workers задаёт процессы загрузки.
--no-augmentation отключает аугментации; --no-mirror отключает только отражение.
Отражение стоит отключать, если оно меняет смысл выбранных жестов.
--patience 0 отключает раннюю остановку. --seed задаёт воспроизводимое начальное состояние;
побитовая воспроизводимость GPU-операций не гарантируется.

По умолчанию train.py строит 3 StratifiedGroupKFold фолда только внутри train
и использует фолд 0 как валидацию. Менять через --folds и --fold.
Один человек (user_id из имени .npy) не попадает одновременно в обучение и валидацию.
Аугментации и балансировка применяются только к обучающей части каждого фолда.
Фолды сохраняются в splits.json. Если классов слишком мало для групповой валидации,
скрипт выдаёт ошибку, а не подменяет её случайным разбиением клипов.

При групповой стратификации отдельный редкий класс может отсутствовать в validation:
скрипт перечисляет такие классы. Macro-F1 считается по полному словарю с нулевым вкладом
от отсутствующих классов; balanced accuracy — по классам с поддержкой.
Для сравнения конфигураций разбиение остаётся одинаковым.

Новый запуск сохраняется в runs/<architecture>_<timestamp>/.
Исходная models/sign_gru.pt не перезаписывается. --run-dir позволяет задать новую папку,
--out — дополнительно скопировать checkpoint в новый файл; существующий --out защищён.
Структура checkpoint содержит архитектуру, параметры, классы и длину последовательности.
Загрузчик также поддерживает старую модель SignGRU.

## Перебор с кросс-валидацией

Сначала можно сохранить только план без обучения:

```bash
python src/translator/experiments.py --trials 12 --folds 3 --dry-run
```

Сравнить 12 конфигураций на трёх одинаковых фолдах, затем переобучить победителя
на всём train:

```bash
python src/translator/experiments.py --trials 12 --folds 3 --epochs 80 \
  --patience 12 --device cuda --refit
```

Это до 36 обучений фолдов плюс финальное обучение. Первые три конфигурации
встроенного random search сравнивают архитектуры при общих начальных настройках.
Остальные случайно выбирают параметры, функцию потерь, балансировку, mixup и scheduler.
Параметры задаются в начале experiments.py; --seed фиксирует поиск и разбиение.
Оптимизатор и модель инициализируются заново в каждом фолде.

Для более узкого поиска используй configs/search-small.json — это 12 комбинаций
архитектуры, hidden и lr:

```bash
python src/translator/experiments.py --space configs/search-small.json \
  --method grid --trials 12 --folds 3 --epochs 80 --device cuda --refit
```

JSON пространства: ключи как в Config, значения — непустые списки.
--method grid перебирает всё пространство; --trials задаёт защитный максимум
и не допускает незаметного запуска огромной сетки.
--config принимает базовую конфигурацию; параметры из --space имеют приоритет.
Для своих архитектур/heads комбинации hidden должны делиться на heads.

Победитель выбирается по среднему macro-F1 фолдов. Учитываются все классы,
поэтому многочисленный no_event не определяет результат в одиночку.
Для final refit число эпох — медиана лучших эпох фолдов победителя:
решение принимается по CV, без использования test.

В папке runs/search_<timestamp>/:
- plan.json, splits.json — параметры запуска, список файлов и фолды;
- trial_*/fold_*/best.pt, history.json, result.json — checkpoint, история, метрики;
- results.json, leaderboard.csv — результаты и рейтинг завершённых конфигураций;
- best_config.json, winner.json — выбранные параметры, метрики, рекомендуемые эпохи;
- final/best.pt — модель на всём train, если указан --refit.

Результаты записываются после каждого фолда/конфигурации. Автоматического продолжения
незавершённого запуска пока нет; --run-dir требует новую папку.

## Итоговый test и повторное обучение

Официальный test не читается для подбора параметров и остановки обучения.
Его можно проверить явно после фиксации победителя:

```bash
python src/translator/evaluate.py --model runs/search_<timestamp>/final/best.pt --device cuda
```

Или добавить --test к experiments.py --refit. Отчёт test_metrics.json содержит
accuracy, macro-F1, balanced accuracy, метрики каждого класса и confusion matrix.
Также выводится пересечение людей между официальными train/test:
такой test может содержать знакомых людей и отличается от signer-disjoint CV.

Старый train.py ранее использовал этот же test для выбора эпохи,
поэтому исторические результаты не следует считать независимой оценкой.
CV после подбора параметров тоже служит выбору модели, а не заменяет итоговый test.
Сравнивай модели на одном наборе классов: 24 и 72 класса — разные задачи.

Повторное обучение выбранной конфигурации с другим seed и фиксированным числом эпох:

```bash
python src/translator/train.py --config runs/search_<timestamp>/best_config.json \
  --refit --epochs 60 --seed 42 --device cuda
```

## Проверки кода

```bash
python -m unittest discover -s tests -v
```

Тесты используют искусственные данные и проверяют групповые фолды,
градиенты архитектур, метрики, короткие последовательности, варианты обучения
и совместимость checkpoint. Полноценное обучение на Slovo запускается вручную.

Методология:
- [StratifiedGroupKFold и групповая валидация](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data).
- [TransformerEncoderLayer](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html).
