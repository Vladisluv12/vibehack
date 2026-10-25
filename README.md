# Second Brain

Локальная база знаний на Markdown: источник истины — файлы `.md` (открываются в
Obsidian), поиск и граф — пересоздаваемый индекс SQLite FTS5. У каждой записи есть
происхождение (`source`, `source_hash`) и статус проверки:

- `unreviewed` — импортировано, никто не проверял;
- `proposed` — предложено человеком или агентом;
- `reviewed` — проверено человеком. Агент не может выставить этот статус.

Runtime — только стандартная библиотека Python 3.11+. Сеть не нужна.

## Быстрый старт

```bash
make test
make demo          # копирует examples/demo-vault в var/ и открывает панель
```

Панель: http://127.0.0.1:8765/ (слушает только localhost, API защищён токеном страницы).

## CLI

```bash
python3 -m second_brain --vault PATH --state PATH sync
python3 -m second_brain --vault PATH search 'запрос'
python3 -m second_brain --vault PATH read <id>
python3 -m second_brain --vault PATH import file.md --split   # внешний источник → unreviewed
python3 -m second_brain --vault PATH backup
python3 -m second_brain restore backup.zip new-empty-dir
python3 -m second_brain --vault PATH serve --port 8765
python3 -m second_brain --vault PATH mcp                       # нужен пакет fastmcp
```

`--vault` и `--state` можно задать через `SECOND_BRAIN_VAULT` / `SECOND_BRAIN_STATE`.
State (индекс, резервные копии) держите вне vault.

## MCP для агентов

`mcp` поднимает STDIO-шлюз с четырьмя инструментами: `search_knowledge`,
`get_document`, `get_related`, `create_draft`. Удаления, правки существующих заметок
и повышение статуса через него недоступны; черновики агента всегда `proposed`.

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

`second_brain/static/` — готовая сборка панели (three.js, 3d-force-graph; лицензии в
`THIRD_PARTY_NOTICES.md`). Исходники в `frontend/`; пересборка требует сети:
`npm install && npm run build`. Лента активности агентов и метрики проектов в
этой версии скрыты, а их API отвечают пустыми снимками.

## Происхождение

Модуль выделен из локального проекта agent-control (оркестратор агентов). Здесь
остались только хранилище, индекс, панель и MCP-шлюз; очередь задач, воркеры и
импорты истории агентов не входят.
