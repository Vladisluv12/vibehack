"""Generate examples/showcase-vault: a synthetic, densely linked team vault for demos.

Deterministic output (fixed IDs and dates), stdlib only:
    python examples/build_showcase.py
"""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil

OUT = Path(__file__).resolve().parent / 'showcase-vault'
FOLDERS = {k: k.capitalize() + 's' for k in ('project', 'research', 'source', 'decision', 'session', 'concept', 'note')}

# key, kind, review, title, source, relations [(relation, key)], body
NOTES = [
    # Projects
    ('core', 'project', 'reviewed', 'Second Brain', '', [],
     'Локальная база знаний на Markdown для команды и AI-агентов. Файлы — источник истины, индекс пересоздаётся в любой момент.'),
    ('agents', 'project', 'reviewed', 'Agent Control', '', [('depends_on', 'core')],
     'Оркестрация агентов: очередь задач, воркеры, лента вызовов. Агенты читают общую память и предлагают черновики.'),
    ('search', 'project', 'reviewed', 'Поиск и ранжирование', '', [('depends_on', 'core')],
     'Как найти нужную заметку за доли секунды: лексический поиск, ранжирование, морфология и гибридные подходы.'),
    ('security', 'project', 'reviewed', 'Безопасность', '', [('relates_to', 'core')],
     'Локальный сервер, токены, редакция секретов и защита от недоверенного содержимого.'),
    ('onboarding', 'project', 'reviewed', 'Онбординг', '', [('relates_to', 'core')],
     'Путь новой команды от установки до первой полезной находки в памяти.'),
    ('gtm', 'project', 'reviewed', 'Go-to-market', '', [('relates_to', 'core')],
     'Позиционирование, цены, пилоты и истории для презентаций.'),
    # Shared concepts
    ('md', 'concept', 'reviewed', 'Markdown — источник истины', '', [('relates_to', 'wiki')],
     'Заметки — обычные `.md` файлы. Их можно открыть в любом редакторе, положить в git и пережить смену инструмента.'),
    ('prov', 'concept', 'reviewed', 'Происхождение данных', '', [('relates_to', 'review')],
     'У каждой записи есть `source` и `source_hash`: откуда она пришла и в каком виде.'),
    ('review', 'concept', 'reviewed', 'Статусы проверки', '', [],
     'Три состояния:\n\n- **unreviewed** — импортировано, никто не смотрел;\n- **proposed** — предложено человеком или агентом;\n- **reviewed** — проверено человеком.'),
    ('bm25', 'concept', 'reviewed', 'BM25', '', [('relates_to', 'fts')],
     'Классическая функция ранжирования: учитывает частоту термина, его редкость и длину документа.'),
    ('fts', 'concept', 'reviewed', 'FTS5', '', [],
     'Полнотекстовый индекс SQLite. Встроен в Python, не требует отдельного сервера.'),
    ('mcp', 'concept', 'reviewed', 'MCP', '', [],
     'Model Context Protocol — стандартный способ дать агенту инструменты и данные.'),
    ('graph', 'concept', 'reviewed', 'Граф знаний', '', [('relates_to', 'wiki')],
     'Заметки — узлы, явные ссылки между ними — рёбра. Граф показывает, что с чем связано и что осталось сиротой.'),
    ('rag', 'concept', 'reviewed', 'RAG', '', [('relates_to', 'bm25'), ('relates_to', 'embed')],
     'Retrieval-augmented generation: модель сначала находит контекст, потом отвечает.'),
    ('embed', 'concept', 'proposed', 'Эмбеддинги', '', [],
     'Векторные представления текста для поиска по смыслу, а не по словам.'),
    ('redact', 'concept', 'reviewed', 'Редакция секретов', '', [],
     'Токены, ключи и пароли вырезаются из текста до записи в vault.'),
    ('local', 'concept', 'reviewed', 'Local-first', '', [('relates_to', 'md')],
     'Данные живут на машине пользователя. Сеть не нужна ни для чтения, ни для поиска.'),
    ('hitl', 'concept', 'reviewed', 'Human-in-the-loop', '', [('relates_to', 'review')],
     'Агент предлагает — человек утверждает. Ни одна запись не становится «проверенной» без человека.'),
    ('wiki', 'concept', 'reviewed', 'Вики-ссылки', '', [],
     'Название заметки в двойных квадратных скобках превращается в связь. Синтаксис совместим с Obsidian.'),
    # Second Brain
    ('sqlite', 'decision', 'reviewed', 'SQLite как индекс', '', [('belongs_to', 'core'), ('depends_on', 'fts'), ('relates_to', 'md')],
     'Индекс можно удалить и пересобрать из файлов. Поэтому он никогда не становится вторым источником истины.'),
    ('obsidian', 'decision', 'reviewed', 'Совместимость с Obsidian', '', [('belongs_to', 'core'), ('relates_to', 'wiki'), ('relates_to', 'md')],
     'Vault открывается в Obsidian как есть: те же папки, frontmatter и вики-ссылки.'),
    ('versions', 'decision', 'reviewed', 'Версия при каждой правке', '', [('belongs_to', 'core'), ('relates_to', 'prov')],
     'Перед записью старое содержимое сохраняется в `Versions/<id>/<hash>.md`. Откат — простое копирование файла.'),
    ('formats', 'research', 'reviewed', 'Форматы хранения', '', [('belongs_to', 'core'), ('relates_to', 'sqlite'), ('relates_to', 'md')],
     'Сравнили Markdown, JSON и базу данных. Markdown выигрывает по читаемости, переносимости и diff в git.'),
    ('obsidian_docs', 'source', 'unreviewed', 'Obsidian: внутренние ссылки', 'https://help.obsidian.md/links', [('relates_to', 'wiki'), ('relates_to', 'obsidian')],
     'Документация Obsidian о синтаксисе внутренних ссылок, алиасах и ссылках на заголовки.'),
    ('backup', 'note', 'reviewed', 'Бэкап с манифестом', '', [('belongs_to', 'core'), ('relates_to', 'versions')],
     'Архив хранит хеш каждого файла. Восстановление проверяет все суммы до первой записи на диск.'),
    ('incsync', 'note', 'proposed', 'Инкрементальный sync', 'agent:second-brain-mcp', [('belongs_to', 'core'), ('relates_to', 'sqlite'), ('relates_to', 'scale')],
     'Предложение агента: пересобирать только файлы с изменившимся mtime или хешем.'),
    ('arch_session', 'session', 'unreviewed', 'Сессия: архитектура ядра', 'agent:claude-code', [('relates_to', 'core'), ('relates_to', 'sqlite'), ('relates_to', 'versions')],
     'Агент разобрал модуль хранения и предложил разделить запись, индекс и граф.'),
    ('conflicts', 'note', 'reviewed', 'Конфликты правок', '', [('belongs_to', 'core'), ('relates_to', 'versions')],
     'Редактор отправляет хеш версии, которую видел. Если файл изменился снаружи, сохранение отклоняется.'),
    ('scale', 'research', 'reviewed', 'Нагрузка: 10 000 заметок', '', [('belongs_to', 'core'), ('relates_to', 'sqlite')],
     'Полная пересборка индекса укладывается в секунды. Узкое место — чтение файлов, а не SQLite.'),
    ('crossplat', 'decision', 'reviewed', 'Linux, macOS и Windows', '', [('belongs_to', 'core'), ('relates_to', 'conflicts')],
     'Файлы пишутся байт в байт, блокировки — через `flock` или `msvcrt`.'),
    # Agent Control
    ('drafts', 'decision', 'reviewed', 'Агент пишет только черновики', '', [('belongs_to', 'agents'), ('relates_to', 'review'), ('relates_to', 'hitl')],
     'Все записи агента получают статус `proposed`. Повысить статус может только человек.'),
    ('mcp_gw', 'decision', 'reviewed', 'MCP-шлюз: четыре инструмента', '', [('belongs_to', 'agents'), ('depends_on', 'mcp'), ('relates_to', 'drafts')],
     '`search_knowledge`, `get_document`, `get_related`, `create_draft`. Удалений и правок нет.'),
    ('queue', 'note', 'reviewed', 'Очередь задач', '', [('belongs_to', 'agents')],
     'Задачи ставятся из панели или CLI и выполняются воркерами по одной на проект.'),
    ('workers', 'note', 'reviewed', 'Воркеры и ретраи', '', [('belongs_to', 'agents'), ('relates_to', 'queue')],
     'Упавшая задача повторяется с экспоненциальной паузой и сохраняет журнал попыток.'),
    ('feed', 'note', 'reviewed', 'Лента вызовов', '', [('belongs_to', 'agents'), ('relates_to', 'queue')],
     'Каждый вызов инструмента и обращение к памяти видны в реальном времени.'),
    ('hallu', 'research', 'reviewed', 'Галлюцинации в памяти агентов', '', [('belongs_to', 'agents'), ('relates_to', 'review'), ('relates_to', 'prov'), ('relates_to', 'drafts')],
     'Если агент пишет в память без проверки, ошибки начинают цитироваться как факты. Статусы проверки разрывают этот цикл.'),
    ('mcp_spec', 'source', 'unreviewed', 'Спецификация MCP', 'https://modelcontextprotocol.io', [('relates_to', 'mcp'), ('relates_to', 'mcp_gw')],
     'Официальная спецификация протокола: инструменты, ресурсы, транспорт STDIO.'),
    ('tickets_session', 'session', 'unreviewed', 'Сессия: агент разбирает тикеты', 'agent:codex', [('relates_to', 'agents'), ('relates_to', 'queue')],
     'Агент прошёлся по очереди, нашёл три дубликата и предложил объединить их.'),
    ('review_session', 'session', 'reviewed', 'Сессия: ревью черновиков', '', [('relates_to', 'drafts'), ('relates_to', 'review')],
     'Команда разобрала 14 черновиков агента: 9 приняты, 3 отправлены на доработку, 2 удалены.'),
    ('citations', 'note', 'reviewed', 'Цитирование источников', '', [('belongs_to', 'agents'), ('relates_to', 'prov'), ('relates_to', 'mcp_gw')],
     'Черновик агента обязан ссылаться на ID заметок, из которых он собран.'),
    ('autolinks', 'note', 'proposed', 'Агент предлагает связи', 'agent:second-brain-mcp', [('belongs_to', 'agents'), ('relates_to', 'graph'), ('relates_to', 'hitl')],
     'Агент находит заметки о похожем и предлагает связь. Человек принимает её одним кликом.'),
    ('evals', 'research', 'proposed', 'Оценка качества ответов', 'agent:second-brain-mcp', [('belongs_to', 'agents'), ('relates_to', 'hallu'), ('relates_to', 'rag')],
     'Набор вопросов с эталонными ответами и ссылками. Метрика — доля ответов с верным источником.'),
    # Search
    ('bm25_default', 'decision', 'reviewed', 'BM25 по умолчанию', '', [('belongs_to', 'search'), ('depends_on', 'bm25'), ('depends_on', 'fts')],
     'Лексический поиск предсказуем, объясним и работает без моделей.'),
    ('morph', 'research', 'reviewed', 'Русская морфология', '', [('belongs_to', 'search'), ('relates_to', 'fts'), ('relates_to', 'bm25_default')],
     '«решение» не находит «решения». Нужен стемминг или лемматизация на этапе индексации.'),
    ('stemmer', 'note', 'proposed', 'Стеммер Snowball', 'agent:second-brain-mcp', [('belongs_to', 'search'), ('derived_from', 'morph')],
     'Лёгкий алгоритмический стеммер без зависимостей, около 150 строк.'),
    ('hybrid', 'research', 'reviewed', 'Гибридный поиск', '', [('belongs_to', 'search'), ('relates_to', 'embed'), ('relates_to', 'bm25'), ('relates_to', 'rag')],
     'Объединение лексической и векторной выдачи через reciprocal rank fusion.'),
    ('fts_docs', 'source', 'unreviewed', 'SQLite FTS5', 'https://sqlite.org/fts5.html', [('relates_to', 'fts'), ('relates_to', 'bm25_default')],
     'Документация FTS5: токенизаторы, функции `bm25()` и `snippet()`.'),
    ('bm25_wiki', 'source', 'unreviewed', 'Okapi BM25', 'https://en.wikipedia.org/wiki/Okapi_BM25', [('relates_to', 'bm25')],
     'Формула и параметры k1 и b.'),
    ('title_weight', 'decision', 'reviewed', 'Заголовок весит ×5', '', [('belongs_to', 'search'), ('relates_to', 'bm25_default')],
     'Совпадение в заголовке почти всегда важнее, чем упоминание в длинном тексте.'),
    ('snippets', 'note', 'reviewed', 'Сниппеты в выдаче', '', [('belongs_to', 'search'), ('relates_to', 'bm25_default')],
     'Под каждым результатом — фрагмент текста вокруг совпадения.'),
    ('typos', 'note', 'proposed', 'Опечатки через триграммы', '', [('belongs_to', 'search'), ('relates_to', 'morph')],
     'Триграммный индекс находит «сихронизацию» по запросу «синхронизация».'),
    ('ranking_session', 'session', 'unreviewed', 'Сессия: тюнинг ранжирования', 'agent:claude-code', [('relates_to', 'search'), ('relates_to', 'title_weight')],
     'Агент прогнал 40 запросов и сравнил выдачу до и после изменения весов.'),
    ('local_embed', 'research', 'proposed', 'Локальные эмбеддинги', '', [('belongs_to', 'search'), ('relates_to', 'embed'), ('relates_to', 'local'), ('relates_to', 'hybrid')],
     'Небольшая модель на CPU даёт поиск по смыслу без отправки данных наружу.'),
    # Security
    ('localhost', 'decision', 'reviewed', 'Только localhost и токен', '', [('belongs_to', 'security'), ('relates_to', 'local')],
     'Сервер слушает 127.0.0.1, проверяет Host и Origin. API доступен только со страницы с токеном.'),
    ('redact_input', 'decision', 'reviewed', 'Редакция на входе', '', [('belongs_to', 'security'), ('depends_on', 'redact')],
     'Секреты вырезаются при импорте, создании и показе заметки.'),
    ('paths', 'note', 'reviewed', 'Защита от обхода путей', '', [('belongs_to', 'security'), ('relates_to', 'crossplat')],
     'Отклоняются абсолютные пути, `..`, скрытые папки, симлинки и пути с диском Windows.'),
    ('extlinks', 'note', 'reviewed', 'Внешний текст не создаёт связей', '', [('belongs_to', 'security'), ('relates_to', 'prov'), ('relates_to', 'wiki')],
     'Вики-ссылки внутри импортированного текста — это данные, а не утверждения графа.'),
    ('threats', 'research', 'reviewed', 'Модель угроз', '', [('belongs_to', 'security'), ('relates_to', 'localhost'), ('relates_to', 'redact_input'), ('relates_to', 'paths'), ('relates_to', 'extlinks')],
     'Недоверенные источники, соседние сайты в браузере, утечки секретов и подмена графа.'),
    ('owasp', 'source', 'unreviewed', 'OWASP: Path Traversal', 'https://owasp.org/www-community/attacks/Path_Traversal', [('relates_to', 'paths')],
     'Классы атак на обход каталогов и способы защиты.'),
    ('csp', 'note', 'reviewed', 'CSP и проверка Origin', '', [('belongs_to', 'security'), ('relates_to', 'localhost')],
     'Content-Security-Policy запрещает сторонние скрипты и встраивание страницы во фреймы.'),
    ('audit_session', 'session', 'unreviewed', 'Сессия: аудит безопасности', 'agent:claude-code', [('relates_to', 'security'), ('relates_to', 'threats')],
     'Агент проверил обработку путей и нашёл обход через пути с диском на Windows. Исправлено.'),
    ('supply', 'note', 'reviewed', 'Пакеты не моложе 14 дней', '', [('belongs_to', 'security'), ('relates_to', 'csp')],
     'Сборка не ставит свежие версии зависимостей: защита от захваченных пакетов.'),
    ('injection', 'note', 'proposed', 'Защита от prompt injection', 'agent:second-brain-mcp', [('belongs_to', 'security'), ('relates_to', 'hallu'), ('relates_to', 'extlinks')],
     'Содержимое заметок передаётся агенту как данные с явной пометкой «не инструкции».'),
    # Onboarding
    ('five_min', 'note', 'reviewed', 'Запуск за 5 минут', '', [('belongs_to', 'onboarding'), ('relates_to', 'local')],
     'Одна команда: скопировать демо и открыть панель. Ни регистрации, ни облака.'),
    ('stdlib', 'decision', 'reviewed', 'Только стандартная библиотека', '', [('belongs_to', 'onboarding'), ('relates_to', 'five_min')],
     'Runtime без зависимостей: нечего ставить, нечему ломаться.'),
    ('imports', 'note', 'reviewed', 'Импорт Markdown и Notion', '', [('belongs_to', 'onboarding'), ('relates_to', 'prov')],
     'Импортированные записи получают статус `unreviewed` и ссылку на источник.'),
    ('interviews', 'research', 'reviewed', 'Интервью с командами', '', [('belongs_to', 'onboarding'), ('relates_to', 'imports')],
     'Главная боль — знания разбросаны по чатам, а агенты не знают контекста команды.'),
    ('persona', 'note', 'reviewed', 'Персона: техлид', '', [('belongs_to', 'onboarding'), ('relates_to', 'interviews')],
     'Отвечает за решения команды и хочет, чтобы агенты опирались на них, а не придумывали.'),
    ('offline_demo', 'note', 'reviewed', 'Демо без сети', '', [('belongs_to', 'onboarding'), ('relates_to', 'stdlib')],
     'Перед любой демонстрацией проверяем запуск с выключенным Wi-Fi.'),
    ('first_run', 'session', 'unreviewed', 'Сессия: первый запуск', '', [('relates_to', 'onboarding'), ('relates_to', 'five_min')],
     'Запись созвона: команда подключила свой vault и нашла забытое решение за первые 10 минут.'),
    ('templates', 'note', 'proposed', 'Шаблоны заметок', 'agent:second-brain-mcp', [('belongs_to', 'onboarding'), ('relates_to', 'imports')],
     'Готовые шаблоны для решений, исследований и ретро.'),
    ('obsidian_users', 'note', 'reviewed', 'Пользователи Obsidian', '', [('belongs_to', 'onboarding'), ('relates_to', 'obsidian'), ('relates_to', 'persona')],
     'Им не нужно ничего переносить: открываем существующий vault.'),
    # Go-to-market
    ('market', 'note', 'reviewed', 'Рынок: память для агентов', '', [('belongs_to', 'gtm'), ('relates_to', 'rag'), ('relates_to', 'interviews')],
     'Каждая команда с AI-агентами упирается в одно: агенту нужен контекст, которому можно доверять.'),
    ('competitors', 'research', 'reviewed', 'Конкуренты', '', [('belongs_to', 'gtm'), ('relates_to', 'market'), ('relates_to', 'local')],
     'Облачные вики не дают происхождения и статусов проверки. Векторные базы не дают человеку читаемый источник.'),
    ('pricing', 'decision', 'proposed', 'Open core', '', [('belongs_to', 'gtm'), ('relates_to', 'competitors')],
     'Ядро бесплатно и открыто. Платно — командная синхронизация, роли и аудит.'),
    ('pitch', 'note', 'reviewed', 'Питч: память, которой можно доверять', '', [('belongs_to', 'gtm'), ('relates_to', 'review'), ('relates_to', 'prov'), ('relates_to', 'market')],
     'Агенты уже пишут код. Им не хватает памяти команды — проверенной, с источниками и под контролем людей.'),
    ('pilot_metrics', 'note', 'proposed', 'Метрики пилотов', '', [('belongs_to', 'gtm'), ('relates_to', 'interviews'), ('relates_to', 'pitch')],
     'Что меряем: время до первой находки, доля ответов агента со ссылкой на проверенную заметку.'),
    ('trends', 'research', 'proposed', 'Тренды: агенты в командах', 'agent:second-brain-mcp', [('belongs_to', 'gtm'), ('relates_to', 'market'), ('relates_to', 'mcp')],
     'MCP становится стандартом подключения инструментов, а память — следующим узким местом.'),
    ('demo_day', 'session', 'unreviewed', 'Сессия: подготовка к демо-дню', 'agent:claude-code', [('relates_to', 'gtm'), ('relates_to', 'pitch')],
     'Собрали сценарий демо: поиск, граф, черновик агента и его проверка человеком.'),
    ('oss', 'decision', 'reviewed', 'Открытое ядро на GitHub', '', [('belongs_to', 'gtm'), ('relates_to', 'pricing')],
     'Открытый код снимает вопрос доверия к локальному инструменту.'),
]

# Agent layer: many short task logs and drafts, tag-heavy and densely cross-linked,
# so the agent space is larger and noisier than the human one.
AGENT_TOPICS = {
    'core': ('core', ['пересборка индекса', 'дубли ID', 'миграция Versions', 'бэкап vault', 'переводы строк', 'блокировки файлов']),
    'agents': ('agents', ['ретраи воркеров', 'лента вызовов', 'лимиты MCP', 'приоритеты очереди', 'цитаты в черновиках', 'разбор черновиков']),
    'search': ('search', ['веса заголовков', 'сниппеты', 'стемминг', 'триграммы', 'гибридная выдача', 'замер качества поиска']),
    'security': ('security', ['аудит путей', 'проверка Origin', 'вырезание токенов', 'обновление зависимостей', 'фильтр инструкций', 'заголовки CSP']),
    'onboarding': ('onboarding', ['офлайн-демо', 'шаблоны заметок', 'импорт из Notion', 'сценарий первого запуска', 'справка CLI', 'частые вопросы']),
    'gtm': ('gtm', ['таблица конкурентов', 'слайды питча', 'сбор метрик пилота', 'текст лендинга', 'тарифы', 'репетиция демо']),
}
AGENTS = ['agent:claude-code', 'agent:codex', 'agent:second-brain-mcp']
TAGS = {}


def agent_notes():
    out, projects = [], list(AGENT_TOPICS)
    for p, (project, topics) in enumerate(AGENT_TOPICS.values()):
        for t, topic in enumerate(topics):
            n = p * len(topics) + t
            log, draft = f'a_log_{project}_{t}', f'a_draft_{project}_{t}'
            next_log = f'a_log_{project}_{(t + 1) % len(topics)}'
            other = f'a_draft_{projects[(p + 1) % len(projects)]}_{t}'
            tags = ['agent', project, topic.split()[0], ['auto', 'todo', 'review-needed'][n % 3]]
            TAGS[log] = TAGS[draft] = tags
            hashtags = ' '.join('#' + tag.replace(' ', '-') for tag in tags)
            agent = AGENTS[n % len(AGENTS)]
            out.append((log, 'session', 'unreviewed', f'Лог: {topic}', agent,
                        [('relates_to', project), ('relates_to', next_log), ('relates_to', f'a_log_{project}_{(t + 3) % len(topics)}')],
                        f'Агент выполнил задачу «{topic}»: прочитал связанные заметки, внёс правки и оставил черновик на проверку.\n\n{hashtags}'))
            out.append((draft, 'note', 'proposed', f'Черновик: {topic}', agent,
                        [('derived_from', log), ('relates_to', project), ('relates_to', f'a_draft_{project}_{(t + 2) % len(topics)}')]
                        + ([('relates_to', other)] if t % 2 == 0 else []),
                        f'Предложение агента по теме «{topic}». Требует проверки человеком перед тем, как стать частью памяти.\n\n{hashtags}'))
    return out


NOTES += agent_notes()
# Humans tag sparingly: one project tag on notes that belong to a project.
PROJECT_TAGS = {'core': 'ядро', 'agents': 'агенты', 'search': 'поиск', 'security': 'безопасность', 'onboarding': 'онбординг', 'gtm': 'gtm'}
for _key, _kind, _review, _title, _source, _relations, _body in NOTES:
    for _rel, _target in _relations:
        if _rel == 'belongs_to' and _key not in TAGS:
            TAGS[_key] = [PROJECT_TAGS[_target]]


def ident(key):
    return hashlib.sha256(('showcase:' + key).encode()).hexdigest()[:32]


def main():
    keys = [n[0] for n in NOTES]
    assert len(keys) == len(set(keys)), 'duplicate keys'
    titles = [n[3] for n in NOTES]
    assert len(titles) == len(set(titles)), 'duplicate titles'
    if OUT.exists():
        shutil.rmtree(OUT)
    start = datetime(2026, 6, 2, 9, 0, tzinfo=timezone.utc)
    for i, (key, kind, review, title, source, relations, body) in enumerate(NOTES):
        for _, target in relations:
            assert target in keys, (key, target)
        meta = {'id': ident(key), 'title': title, 'type': kind, 'permalink': f'{kind}/{ident(key)}', 'review': review,
                'created': (start + timedelta(hours=37 * i)).strftime('%Y-%m-%dT%H:%M:%SZ'),
                'source': source, 'source_hash': '', **({'tags': TAGS[key]} if key in TAGS else {})}
        text = '---\n' + ''.join(f'{k}: {json.dumps(v, ensure_ascii=False)}\n' for k, v in meta.items()) + '---\n\n' + body + '\n'
        if relations:
            text += '\n## Relations\n' + ''.join(f'- {rel} [[{ident(target)}]]\n' for rel, target in relations)
        path = OUT / FOLDERS[kind] / f'{ident(key)}.md'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8', newline='\n')
    print(f'{len(NOTES)} notes, {sum(len(n[5]) for n in NOTES)} relations -> {OUT}')


if __name__ == '__main__':
    main()
