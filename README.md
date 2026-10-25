<div align="center">

<img src="docs/media/logo.png" width="120" alt="Жестик"/>

# Жестик 🤟

### Переводчик с русского жестового языка в текст

Запиши кружок в Telegram или видео на сайте — нейросеть найдёт руки, лицо и позу<br/>
и переведёт жесты РЖЯ в текст за пару секунд.

[![Telegram](https://img.shields.io/badge/Telegram-@zhestik__bot-26A5E4?style=for-the-badge&logo=telegram&logoColor=white)](https://t.me/zhestik_bot)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Holistic-0097A7?style=for-the-badge&logo=google&logoColor=white)](https://github.com/google-ai-edge/mediapipe)

[![aiogram](https://img.shields.io/badge/aiogram-3-2CA5E0?style=flat-square)](https://docs.aiogram.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-web-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Dataset](https://img.shields.io/badge/dataset-Slovo-8A2BE2?style=flat-square)](https://github.com/hukenovs/slovo)
[![Words](https://img.shields.io/badge/словарь-400%20жестов-FF7A59?style=flat-square)](#-словарь)

[**Попробовать бота**](https://t.me/zhestik_bot) · [**Демо**](#-демо) · [**Как это работает**](#-как-это-работает) · [**Быстрый старт**](#-быстрый-старт) · [**Обучение**](#-обучение-модели)

</div>

---

## 🎬 Демо

<div align="center">

<!-- Видео: перетащи .mp4 в редактор README на GitHub — он вставит ссылку вида
     https://github.com/user-attachments/assets/... — и замени строку ниже на неё. -->
https://github.com/user-attachments/assets/REPLACE-WITH-DEMO-VIDEO-ID

</div>

<table>
  <tr>
    <td align="center" width="33%">
      <img src="docs/media/bot-videonote.gif" alt="Кружок в боте" width="260"/><br/>
      <b>1. Записываешь кружок</b><br/>
      <sub>прямо в чате с ботом</sub>
    </td>
    <td align="center" width="33%">
      <img src="docs/media/bot-result.png" alt="Ответ бота" width="260"/><br/>
      <b>2. Получаешь перевод</b><br/>
      <sub>текст + уверенность по каждому слову</sub>
    </td>
    <td align="center" width="33%">
      <img src="docs/media/bot-start.png" alt="Меню бота" width="260"/><br/>
      <b>3. Удобное меню</b><br/>
      <sub>инструкция, словарь, статус модели</sub>
    </td>
  </tr>
</table>

<div align="center">
  <img src="docs/media/site.png" alt="Веб-версия Жестика" width="88%"/><br/>
  <sub>Веб-версия: запись с камеры в браузере, загрузка файла, озвучка перевода, светлая и тёмная тема</sub>
</div>

---

## ✨ Возможности

|  |  |
|---|---|
| 📹 **Кружки и видео** | Бот принимает кружки, видео и видеофайлы до 20 МБ, сайт — запись с веб-камеры и файлы до 50 МБ |
| 🧩 **Фразы, а не только слова** | Длинное видео разбивается скользящим окном, жесты складываются во фразу: *«Привет, я рад»* |
| 🎯 **Уверенность модели** | Для каждого слова — вероятность; если жест не распознан, показываются варианты |
| ⚡ **Работает на CPU** | Модель ~1–2 млн параметров, видеокарта нужна только для быстрого обучения |
| 🌗 **Красивый веб-интерфейс** | Запись с отсчётом 3-2-1, история переводов, озвучка, тёмная тема |
| 🔄 **Горячая замена модели** | Бот и сайт сами подхватывают новую модель после обучения — без перезапуска |
| 🖥️ **Распределённая подготовка данных** | Извлечение точек делится между несколькими ПК и собирается одной командой |

---

## 🧠 Как это работает

Жестик смотрит не на пиксели, а на **движение скелета** — поэтому ему не важны фон, свет, одежда и внешность.

```mermaid
flowchart LR
    A[🎥 Видео<br/>кружок / файл / вебка] --> B[🦴 MediaPipe Holistic<br/>поза · руки · лицо]
    B --> C[🔢 425 признаков<br/>на каждый кадр]
    C --> D[⏱️ 32 кадра<br/>+ скорости движения]
    D --> E[🧠 Нейросеть<br/>GRU / Attention-GRU / Transformer]
    E --> F[💬 Слово]
    F --> G[📝 Фраза]

    style B fill:#6c4cf5,color:#fff,stroke:none
    style E fill:#ff7a59,color:#fff,stroke:none
```

**1. Скелет.** [MediaPipe Holistic](https://github.com/google-ai-edge/mediapipe) — готовая замороженная сеть от Google — находит в каждом кадре 140+ точек.

**2. Признаки.** Точки нормализуются относительно плеч (неважно, где человек стоит и как далеко от камеры) и собираются в вектор из 425 чисел:

| Блок | Что описывает | Размер |
|---|---|:---:|
| 🧍 Поза | плечи, локти, запястья, голова | 75 |
| ✋ Руки относительно тела | *где* находятся кисти | 63 × 2 |
| 🤌 Форма кистей | *какая* форма: кулак, палец, «V» | 63 × 2 |
| 👄 Лицо | губы и брови — мимика несёт смысл | 96 |
| 🚩 Флаги | найдена ли левая / правая рука | 2 |

**3. Классификация.** Клип сжимается до 32 кадров, к координатам добавляются скорости, и нейросеть по последовательности движений определяет жест. Учится только она — MediaPipe используется как готовый экстрактор признаков.

**4. Фраза.** Для длинных видео окно 1.6 с скользит по записи; слово засчитывается, только если несколько окон подряд уверенно согласны.

### Архитектуры

| `--arch` | Устройство |
|---|---|
| `gru` | Linear → BiGRU (2 слоя) → среднее по времени → классификатор. Базовая модель |
| `attn_gru` | Residual Conv1D-блоки → BiGRU → attention / mean / max pooling → MLP |
| `transformer` | Позиционное кодирование → Transformer Encoder → attention / mean pooling |

Аугментации при обучении: отражение (левша ↔ правша), обрезка по времени, поворот, масштаб, шум трекинга, «потеря» руки. Валидация — **signer-disjoint**: один человек не попадает одновременно в обучение и проверку.

---

## 📊 Результаты

> Заполни после финального `evaluate.py` — метрики лежат в `runs/.../test_metrics.json`.

| Словарь | Модель | Accuracy | Macro-F1 | Balanced acc. |
|---|---|:---:|:---:|:---:|
| 24 жеста | `gru` | — | — | — |
| 400 жестов | `attn_gru` | — | — | — |
| 400 жестов | `transformer` | — | — | — |

---

## 📚 Словарь

Сейчас модель учится на **400 жестах** + класс «нет жеста» из датасета [Slovo](https://github.com/hukenovs/slovo) (SberDevices).

<details>
<summary><b>Примеры слов</b></summary>

<br/>

**Приветствия:** привет · пока · добро пожаловать · с днём рождения
**Местоимения:** я · ты · мы · вы · меня · тебя · мой · твой · наш
**Вопросы:** кто · что · когда · почему · сколько
**Люди:** мать · отец · брат · сестра · друг · семья · ребёнок
**Действия:** хотеть · знать · понять · помочь · есть · пить · идти · работать · любить
**Время:** сегодня · завтра · вчера · утро · вечер · понедельник … воскресенье
**Оценки:** хорошо · плохо · очень · рад · вкусный · интересный

Полный список — в боте (кнопка «📝 Список слов»), на сайте или в [`vocab.json`](src/translator/vocab.json).

</details>

---

## 🚀 Быстрый старт

### 1. Установка

```bash
git clone https://github.com/Vladisluv12/vibehack.git
cd vibehack
git checkout translator

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-training.txt
pip install aiogram aiohttp-socks python-decouple fastapi "uvicorn[standard]" python-multipart
```

> PyTorch ставится отдельно: [CUDA-сборка](https://pytorch.org/get-started/locally/) для обучения на GPU или `pip install torch --index-url https://download.pytorch.org/whl/cpu` для CPU.

### 2. Telegram-бот

Создай `.env` в корне проекта:

```env
BOT_TOKEN=123456:ABC-твой-токен-от-BotFather
TG_PROXY=socks5://127.0.0.1:12334   # необязательно: SOCKS-порт VPN-клиента
```

```bash
python src/translator/bot.py --check   # проверить токен и связь без модели
python src/translator/bot.py           # запустить бота
```

### 3. Сайт

```bash
python src/web/app.py                  # → http://localhost:8000
```

> Камера в браузере работает только на `localhost` или по `https`. Для доступа с телефона: `cloudflared tunnel --url http://localhost:8000`.

### 4. Вебка в реальном времени

```bash
python src/translator/realtime.py      # q — выход, c — очистить фразу
```

### 5. Распознать файл из консоли

```bash
python src/translator/recognizer.py path/to/video.mp4
```

---

## 🏋️ Обучение модели

<details open>
<summary><b>Полный цикл: от видео до модели</b></summary>

<br/>

**1. Датасет.** Скачай [Slovo](https://github.com/hukenovs/slovo) и распакуй в `data/slovo/` (`annotations.csv`, `train/`, `test/`).

**2. Словарь.**

```bash
python src/translator/make_vocab.py --words-file src/translator/words_basic.txt --size 400
```

**3. Извлечение точек** — самый долгий шаг (MediaPipe работает только на CPU). Можно распределить между несколькими ПК пропорционально числу ядер:

| ПК | Ядер | Команда |
|---|:---:|---|
| 1 | 16 | `python src/translator/extract_slovo.py --shard 0,1,2,3/8 --workers 14 --max-side 640 --frame-step 2` |
| 2 | 12 | `python src/translator/extract_slovo.py --shard 4,5,6/8 --workers 10 --max-side 640 --frame-step 2` |
| 3 | 4 | `python src/translator/extract_slovo.py --shard 7/8 --workers 3 --max-side 640 --frame-step 2` |

**4. Сборка** результатов на одном ПК:

```bash
python src/translator/merge_kp.py pack                          # на каждом ПК
python src/translator/merge_kp.py unpack data/kp_*.zip          # на главном
python src/translator/merge_kp.py check                         # всё ли на месте
```

**5. Обучение и подбор архитектуры:**

```bash
python src/translator/train.py --arch attn_gru --epochs 100 --patience 15 --device cuda
python src/translator/experiments.py --space configs/search-small.json --method grid --trials 12 --folds 3 --device cuda --refit
python src/translator/evaluate.py --model runs/search_<timestamp>/final/best.pt --device cuda
```

</details>

Подробно про флаги, кросс-валидацию и перебор гиперпараметров — в [**docs/TRAINING.md**](docs/TRAINING.md).

---

## 🗂️ Структура проекта

```
vibehack/
├── src/
│   ├── translator/
│   │   ├── features.py          # словарь, 425 признаков кадра, нормализация
│   │   ├── model.py             # gru · attn_gru · transformer
│   │   ├── extract_slovo.py     # видео → точки (с разделением между ПК)
│   │   ├── merge_kp.py          # сборка точек с нескольких ПК
│   │   ├── make_vocab.py        # сборка словаря → vocab.json
│   │   ├── train.py             # обучение одной конфигурации
│   │   ├── training.py          # данные, аугментации, фолды, метрики
│   │   ├── experiments.py       # перебор конфигураций с кросс-валидацией
│   │   ├── evaluate.py          # финальная проверка на test
│   │   ├── recognizer.py        # распознавание видеофайла (бот + сайт)
│   │   ├── realtime.py          # распознавание с веб-камеры
│   │   └── bot.py               # Telegram-бот @zhestik_bot
│   └── web/
│       ├── app.py               # FastAPI-бэкенд
│       └── static/index.html    # веб-интерфейс
├── configs/                     # пространства поиска гиперпараметров
├── tests/                       # unit-тесты обучения
├── docs/media/                  # скриншоты и демо для README
├── data/                        # датасет и точки (в .gitignore)
└── runs/                        # результаты обучений (в .gitignore)
```

---

## 🛣️ Планы

- [x] Распознавание изолированных жестов по ключевым точкам
- [x] Telegram-бот: кружки, видео, фразы из нескольких жестов
- [x] Веб-версия с записью с камеры
- [x] Расширение словаря до 400 жестов, распределённое извлечение
- [ ] Перевод последовательности жестов в грамматичное русское предложение с помощью LLM
- [ ] Распознавание прямо в браузере (MediaPipe JS + ONNX)
- [ ] Дообучение на собственных записях и обратная связь от пользователей
- [ ] Сравнение с видео-моделями (3D-CNN, VideoMAE)

---

## 💛 Поддержать проект

Жестик бесплатный и делается в свободное время. Если он вам помог — можно поддержать разработку:

**Сбербанк:** `2202 2068 8532 1526`

---

## 🙏 Благодарности

- [**Slovo**](https://github.com/hukenovs/slovo) — открытый датасет русского жестового языка от SberDevices
- [**MediaPipe**](https://github.com/google-ai-edge/mediapipe) — извлечение ключевых точек тела, рук и лица
- [**aiogram**](https://docs.aiogram.dev/) и [**FastAPI**](https://fastapi.tiangolo.com/) — бот и веб-сервер

<div align="center">

<br/>

Сделано с 💜 на хакатоне

**[⬆ Наверх](#жестик-)**

</div>
