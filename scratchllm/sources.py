"""Configured corpus sources and the exact browsing scope of each source."""
from typing import Literal, get_args
from urllib.parse import parse_qs, unquote, urlsplit

SourceHost = Literal['jewish.win', 'soyjak.st', 'swedishwin.com', 'soyjakwiki.org', 'wikipedia.org']
HOSTS = get_args(SourceHost)
WIKI_HOSTS = frozenset({'soyjakwiki.org', 'wikipedia.org'})
WIKIPEDIA_HOSTS = frozenset({'wikipedia.org', 'www.wikipedia.org', 'en.wikipedia.org', 'en.m.wikipedia.org'})
WIKIPEDIA_NAMESPACES = frozenset({
    'talk', 'user', 'user talk', 'wikipedia', 'wikipedia talk', 'file', 'file talk', 'media',
    'mediawiki', 'mediawiki talk', 'template', 'template talk', 'help', 'help talk',
    'category', 'category talk', 'portal', 'portal talk', 'draft', 'draft talk', 'special',
    'module', 'module talk', 'book', 'book talk', 'timedtext', 'timedtext talk',
    'gadget', 'gadget definition',
})


def seed_url(host):
    return 'https://en.wikipedia.org/wiki/Main_Page' if host == 'wikipedia.org' else f'https://{host}/'


def allowed_host(hostname, source):
    return hostname in (WIKIPEDIA_HOSTS if source == 'wikipedia.org' else {source, 'www.' + source})


def wikipedia_article(url):
    """English article URLs only; Main Page is retained for link discovery."""
    parsed = urlsplit(url)
    query = parse_qs(parsed.query, keep_blank_values=True)
    if parsed.path.startswith('/wiki/'):
        title = unquote(parsed.path[6:])
    elif parsed.path == '/w/index.php':
        title = query.get('title', [''])[0]
    else:
        return None
    if not title or title.lstrip(':').split(':', 1)[0].replace('_', ' ').casefold() in WIKIPEDIA_NAMESPACES:
        return None
    if any(action.casefold() not in {'', 'view'} for action in query.get('action', [])):
        return None
    if any(key.casefold() not in {'title', 'action'} for key in query):
        return None  # History/diff/old revisions, edit forms, search, and downloads.
    return title
