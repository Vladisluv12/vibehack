# Second Brain

Локальная база знаний на Markdown: источник истины — файлы `.md` (открываются в
Obsidian), поиск и граф — пересоздаваемый индекс SQLite FTS5. У каждой записи есть
происхождение (`source`, `source_hash`) и статус проверки:

- `unreviewed` — импортировано, никто не проверял;
- `proposed` — предложено человеком или агентом;
- `reviewed` — проверено человеком. Агент не может выставить этот статус.

Runtime — только стандартная библиотека Python 3.11+. Сеть не нужна (кроме явного импорта
с GitHub). Работает на
Linux, macOS и Windows (на Windows вместо `python3` — `python` или `py`).

## Быстрый старт

```bash
make test          # Python; JS-тесты фронтенда: node --test tests/*.mjs
make demo          # копирует examples/demo-vault (8 заметок) в var/ и открывает панель
make showcase      # то же для examples/showcase-vault: 80 связанных заметок для презентаций
```

Панель: http://127.0.0.1:8765/ (слушает только localhost, API защищён токеном страницы).

## CLI

```bash
python3 -m second_brain --vault PATH --state PATH sync
python3 -m second_brain --vault PATH search 'запрос'
python3 -m second_brain --vault PATH read <id>
python3 -m second_brain --vault PATH import file.md --split   # внешний источник → unreviewed
python3 -m second_brain --vault PATH import-github https://github.com/owner/repo
python3 -m second_brain --vault PATH backup
python3 -m second_brain restore backup.zip new-empty-dir
python3 -m second_brain --vault PATH serve --port 8765
python3 -m second_brain --vault PATH mcp                       # нужен пакет fastmcp
```

`--vault` и `--state` можно задать через `SECOND_BRAIN_VAULT` / `SECOND_BRAIN_STATE`.
State (индекс, резервные копии) держите вне vault.

## Импорт с GitHub

Кнопка «GitHub» в панели или `import-github URL` в CLI. Скачивается архив ветки по
умолчанию; в памяти читаются только `.md`-файлы (без скрытых папок, `node_modules`,
`vendor`; до 200 файлов и 20 МБ текста, архив до 50 МБ), на диск ничего не распаковывается.
Репозиторий становится заметкой-проектом, каждый файл — источником `unreviewed` со ссылкой
на GitHub и связью с проектом, поэтому на графе появляется отдельный кластер. Вики-ссылки
внутри импортированного текста связями не становятся. Повторный импорт неизменённых файлов
дубликатов не создаёт. Приватные репозитории — через переменную окружения `GITHUB_TOKEN`.

## MCP для агентов

`mcp` поднимает STDIO-шлюз с четырьмя инструментами: `search_knowledge`,
`get_document`, `get_related`, `create_draft`. Удаления, правки существующих заметок
и повышение статуса через него недоступны; черновики агента всегда `proposed`.

## Плагины (Agent Control)

GUI целиком входит в этот пакет. Блоки агентов (лента вызовов, метрики проекта,
страница «Очередь агентов») заполняет необязательное расширение — подкласс
`second_brain.web.Extension`, переданный в `make_server(..., extension=...)`.
Без расширения панель пишет «Agent Control не подключён», а `/api/extension`
возвращает `{"connected": false}`. Само хранилище об агентах ничего не знает.

```python
class MyAgents(Extension):
    name = 'Agent Control'
    pages = {'/queue': '<html>…</html>'}        # доп. страницы, __CONTROL_TOKEN__ подставится
    def get(self, path, params, open_brain): ... # None → 404
    def post(self, path, data, open_brain): ...  # (status, body) или None
```

## Формат заметки

```markdown
---
id: "32 hex"
title: "Название"
type: "note"            # project, research, source, decision, session, concept, note
permalink: "note/<id>"
review: "proposed"
created: "2026-10-25T09:00:00Z"
source: ""
source_hash: ""
---

Текст.

## Relations
- relates_to [[<id другой заметки>]]
```

## Фронтенд

`second_brain/static/` — готовая сборка панели (force-graph на canvas, ~200 КБ): граф на весь
экран, кластеры (label propagation с затравкой по проектам) с цветом и свечением, дуги связей,
подписи по важности узла. Поверх графа: KPI, поиск по заметкам (Ctrl K; совпадения
подсвечиваются на графе), выдвижной каталог «Ваши знания» с фильтром по типам, инспектор с Markdown и кликабельными `[[ссылками]]`, режимы
«Кластеры/Типы» и «Весь граф/Окружение 1–2 шага», «Демо-тур» по ключевым узлам. Заливка
узла — статус проверки: сплошная — reviewed, полупрозрачная — proposed, пустая — unreviewed.

`examples/showcase-vault` — синтетическая база вымышленной команды; пересобирается
`python examples/build_showcase.py`. Исходники панели в `frontend/`; пересборка требует сети:
`pnpm install && pnpm run build` (версии закреплены в `pnpm-lock.yaml`, пакеты моложе
14 дней не ставятся; сборка воспроизводится побайтно и заново пишет
`THIRD_PARTY_NOTICES.md` по реально вошедшим в бандл пакетам).

## Происхождение

Модуль выделен из локального проекта agent-control (оркестратор агентов). Здесь
хранилище, индекс, вся панель и MCP-шлюз; очередь задач, воркеры и импорты
истории агентов остаются в agent-control, который подключается как расширение.
