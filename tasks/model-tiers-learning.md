# Tasks: model-tiers-learning

Spec: `specs/model-tiers-learning.md`. База: `vibehack@a598515`, ветка от `second-brain`.
Каждый шаг — отдельный коммит с тестом; `make test` зелёный после каждого.

## Фаза 0 — дефекты, мешающие всему остальному
- [x] F3: `create_draft` → `declared_only=true`, `proposed_links` во frontmatter; тест AC1
- [x] F1: GitHub-импорт резолвит SHA коммита один раз, tarball и URL источников по SHA
- [x] F2: версия импорта (`import_run`), новая версия источника `supersedes` старую

## Фаза 1 — фундамент для обоих уровней
- [x] R0: `state/agents.json`, `--agent-model`, `agent_model/agent_tier` в записях; тест AC6
- [x] S7: crash-safe журнал мутаций (intent → commit), откат по actor/op_id; миграция схемы; тест AC2
- [x] S4: CLI `lint` по полному индексу, `--online` отдельно; тест AC3
- [x] S3: fingerprint-дедуп в `create_draft` под `Brain.lock`; тест AC4
- [x] R1.5: шлюз полей `applies_to/evidence/source_ids` для `experience|skill`; тест AC5
- [x] R1.6: `served`/`report_usage`, служебные чтения не считаются
- [x] R1.7: `truncated` и total в `search_knowledge`
- [x] R1.8: `review`/`agent_model` в выдаче, понижение proposed, метка self-referential

## Фаза 2 — измерение (до включения фаз 3–4)
- [x] формат `evals/tasks/*.json` + `tasks.lock`, исключение eval-источников из выдачи
- [x] runner `eval` с fake-агентом, конфигурации A/B/C, метрики, bootstrap; тест AC8
- [x] адаптеры внешних CLI — общий argv JSON (`--agent`, `--llm`), без ключей; готовых файлов под Claude Code/Codex/OpenCode нет
- [ ] первый замер A/B/C на 30 задачах для одной weak и одной frontier модели (нужны реальные задачи)

## Фаза 3 — weak: самоулучшение
- [x] E1: `ingest-run` (CLI), белый список, redact; приём хуков через `Extension.post` не сделан
- [x] E2: сигнатуры ошибок, 5 триггеров, debounce, без петель
- [x] S2: `reviewer.py` reflect → consolidate, только read + `create_draft`, бюджет; тест AC7
- [x] пакет навыков k=3 по `applies_to`, бюджет символов
- [x] канарейка 20% и сравнение на живых прогонах
- [x] frontier→weak: черновик навыка из успешного frontier-прогона, приём по R2

## Фаза 4 — frontier
- [x] правило записи в описании MCP-сервера
- [x] инструмент `reflect(run_id)` по запросу
- [x] S5 предложения связей как `proposed_links` (из текста черновика; автоподбор соседей не сделан)
- [ ] S6 (`review_at`, прогнозы) — отложено до замера фазы 2

## Review
- [ ] security review: журнал, ingest, GitHub-токен в redirect, web-границы (идёт)
- [x] обновить README (новые CLI и MCP-инструменты)
