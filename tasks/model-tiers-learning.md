# Tasks: model-tiers-learning

Spec: `specs/model-tiers-learning.md`. База: `vibehack@a598515`, ветка от `second-brain`.
Каждый шаг — отдельный коммит с тестом; `make test` зелёный после каждого.

## Фаза 0 — дефекты, мешающие всему остальному
- [x] F3: `create_draft` → `declared_only=true`, `proposed_links` во frontmatter; тест AC1
- [x] F1: GitHub-импорт резолвит SHA коммита один раз, tarball и URL источников по SHA
- [x] F2: версия импорта (`import_run`), новая версия источника `supersedes` старую

## Фаза 1 — фундамент для обоих уровней
- [ ] R0: `state/agents.json`, `--agent-model`, `agent_model/agent_tier` в записях; тест AC6
- [ ] S7: crash-safe журнал мутаций (intent → commit), откат по actor/op_id; миграция схемы; тест AC2
- [ ] S4: CLI `lint` по полному индексу, `--online` отдельно; тест AC3
- [ ] S3: fingerprint-дедуп в `create_draft` под `Brain.lock`; тест AC4
- [ ] R1.5: шлюз полей `applies_to/evidence/source_ids` для `experience|skill`; тест AC5
- [ ] R1.6: `served`/`report_usage`, служебные чтения не считаются
- [ ] R1.7: `truncated` и total в `search_knowledge`
- [ ] R1.8: `review`/`agent_model` в выдаче, понижение proposed, метка self-referential

## Фаза 2 — измерение (до включения фаз 3–4)
- [ ] формат `evals/tasks/*.json` + `tasks.lock`, исключение eval-источников из выдачи
- [ ] runner `eval` с fake-агентом, конфигурации A/B/C, метрики, bootstrap; тест AC8
- [ ] адаптеры внешних CLI (Claude Code, Codex, OpenCode) — argv в конфиге, без ключей
- [ ] первый замер A vs B на 30 задачах для одной weak и одной frontier модели

## Фаза 3 — weak: самоулучшение
- [ ] E1: `ingest-run` + приём хуков через `Extension.post`, белый список, redact
- [ ] E2: сигнатуры ошибок, 5 триггеров, debounce, без петель
- [ ] S2: `reviewer.py` reflect → consolidate, только read + `create_draft`, бюджет; тест AC7
- [ ] пакет навыков k=3 по `applies_to`, бюджет символов
- [ ] канарейка 20% и сравнение на живых прогонах
- [ ] frontier→weak: черновик навыка из успешного frontier-прогона, приём по R2

## Фаза 4 — frontier
- [ ] правило записи в описании MCP-сервера
- [ ] инструмент `reflect(run_id)` по запросу
- [ ] S5 предложения связей как `proposed_links`
- [ ] S6 (`review_at`, прогнозы) — только если замер фазы 2 даст основание

## Review
- [ ] security review: журнал, ingest хуков, GitHub-токен в redirect, web-границы
- [ ] обновить README (новые CLI и MCP-инструменты)
