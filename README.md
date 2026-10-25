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

Кнопка «GitHub» в панели или `import-github URL` в CLI. Ветка по умолчанию один раз
разрешается в коммит, и скачивается архив именно этого коммита; в памяти читаются только `.md`-файлы (без скрытых папок, `node_modules`,
`vendor`; до 200 файлов и 20 МБ текста, архив до 50 МБ), на диск ничего не распаковывается.
Репозиторий становится заметкой-проектом, каждый файл — источником `unreviewed` со ссылкой
на GitHub по SHA коммита и связью с проектом, поэтому на графе появляется отдельный
кластер. Каждый импорт — заметка-сессия `owner/repo @ <commit>` со списком файлов. Вики-ссылки
внутри импортированного текста связями не становятся. Неизменённые файлы при повторном
импорте сохраняют свою заметку; изменённый файл получает новую заметку с полем
`supersedes` на прежнюю версию, старая остаётся. Приватные репозитории — через переменную окружения `GITHUB_TOKEN`.

## MCP для агентов

`mcp --agent-model <id>` поднимает STDIO-шлюз: `search_knowledge`, `get_document`,
`get_related`, `create_draft`, `report_usage`, `get_skill_pack`, `reflect`. Удаления, правки
существующих заметок и повышение статуса через него недоступны; черновики агента всегда
`proposed`. Связями черновика становятся только `source_ids`; вики-ссылки и строки вида
`- contradicts [[...]]` в его тексте сохраняются в `proposed_links` и в граф не попадают,
пока человек не проверит заметку и не уберёт `declared_only`.

- Одинаковый черновик второй раз возвращает первую заметку (`dedup: true`).
- `memory_type: skill|experience` требует `source_ids`, `applies_to` (где верно) и `evidence`
  (как проверено). Черновик, опирающийся только на непроверенные заметки, помечается
  `evidence_basis: self-referential`.
- В выдаче непроверенное помечено `unverified: true` и стоит ниже проверенного при равной
  релевантности; материалы оценки (`source: eval:*`) не выдаются; `truncated` — есть ещё.
- `run_id` в запросах и `report_usage` после задачи дают журнал «выдано / помогло».

## Обучение моделей (state/learning.sqlite)

Уровень модели задаёт хост в `state/agents.json`, агент его сменить не может; неизвестная
модель — `frontier`:

```json
{"models": {"qwen3.8-flash": "weak", "claude-opus-5-5": "frontier"}, "default": "frontier"}
```

Механизмы `skills` и `reflection` по умолчанию выключены для обоих уровней. Включает их только
человек (`mechanism --tier weak --name reflection --on --reason "…"`); оценка может их только
выключить. Автоматическая рефлексия разрешена лишь слабым моделям.

- Журнал правок `state/ledger.jsonl`: намерение с fsync до записи файла, фиксация после.
  Без намерения правки нет; незавершённое сверяется по хешу при открытии. Откат:
  `rollback --actor agent:<model>` или `--op-id`; правка, изменённая позже, — конфликт, созданная
  заметка уходит в `Versions/`.
  `restore` журнал не пишет: он создаёт новое хранилище в пустой папке, а журнал нового
  хранилища начинается с первой правки в нём.
- `lint` — проверки без LLM по всему индексу (битые ссылки, `supersedes`, `source_hash`, навыки без
  полей, дубли, сироты); код выхода 1 при ошибках, `--online` — проверка URL источников, только
  публичных адресов (внутренние, loopback и metadata не запрашиваются, в том числе после редиректа).
- Хост передаёт итог запуска агента: `ingest-run --model <id> run.json`; модель называет хост,
  событие с другой моделью отклоняется (поля: `run_id`, `agent_model`, `task`,
  `outcome`, `steps`, `user_correction`, `errors`, `skills_used`, `contradicts`, `summary`; остальное
  отбрасывается, текст вычищается от секретов). Для слабых моделей с включённой рефлексией
  создаются триггеры: повтор ошибки (по хешу сигнатуры), поправка пользователя, длинная процедура,
  противоречие проверенному, провал навыка. Долгий успешный запуск frontier даёт триггер `distill`:
  навык для слабых. `reflect-request` / MCP `reflect` — рефлексия по запросу для любого уровня.
- `review --llm model.json` — ревизор в две стадии (анализ → сверка с навыками), модель — argv JSON,
  промпт в stdin. Пишет только черновики `proposed` от имени `agent:reviewer`.
- `skill-pack` / MCP `get_skill_pack` — до 3 проверенных навыков к задаче, только если слово из их
  `applies_to` есть в задаче (проверка лексическая, `applies_to` виден модели в пакете). Новый навык сначала
  выдаётся в 20% запусков (`canary`); `promote --stage full|off` — решение человека.
- `stale` — заметки, которые выдаются, но ни разу не помогли.
- Оценка: `eval-lock evals/tasks` замораживает задачи; `eval-run --config A|B|C --model … --agent
  agent.json` (A — без навыков, B — навыки не от ревизора, C — все навыки);
  `eval-decide --baseline … --candidate …` — парный bootstrap: `skills` сравнивают A→C,
  `reflection` — B→C. Нет прироста — механизм выключается; есть — только рекомендация человеку.
  Пары проверяются строго: та же модель и уровень, те же задачи с тем же хешем входов.
  Задача: `prompt`, `workspace` (входы агента), `oracle` (проверяющие файлы, копируются в отдельную
  папку только после завершения агента, `{workspace}` подставляется в `check.argv`). Входы — только
  обычные файлы внутри корня задач; `tasks.lock` фиксирует байты и права всех файлов.
  Готовые конфигурации — `examples/learning/`: `eval-agent-{claude,codex,opencode}.json` (агент в
  `eval-run`), `llm-{claude,opencode}.json` (модель ревизора: `review --llm … --llm-model …`),
  `agents.json`. Агента оборачивает `python3 -m second_brain.usage_adapter --format claude|codex|opencode`:
  вывод CLI уходит в stderr, а в stdout — одна строка usage по учёту самого провайдера
  (вход + выход + рассуждения, без чтения кэша). Запускайте с `PYTHONPATH=<репозиторий>`.
  Без адаптера токены агент сообщает сам строкой `{"usage":{"tokens":N}}` — для честной стоимости
  используйте адаптер выше; агент работает без
  песочницы ОС, поэтому оставленный им фоновый процесс теоретически может дотянуться до oracle.

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
