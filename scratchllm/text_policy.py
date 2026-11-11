"""Shared text-only corpus policy; independent of browser and tokenizer code."""
import re
from urllib.parse import parse_qsl, unquote, urlsplit
from .sources import WIKIPEDIA_HOSTS, wikipedia_article

EXTRACTION_VERSION = "article-post-text-v4"

# A media link can be a navigation, an XHR, or a browser subresource. Inspect its
# URL as well as Playwright's resource type, including encoded/query filenames.
ATTACHMENT_SUFFIXES = frozenset("""
mp3 mp2 mp1 wav wave ogg oga opus flac aac m4a m4b aif aiff wma amr au mid midi
mp4 m4v webm mkv avi mov qt mpeg mpg mpe ogv ogm wmv flv 3gp ts m3u m3u8 mpd pls
jpg jpeg jfif png gif webp avif bmp dib tif tiff svg svgz ico heic heif apng jxl psd
pdf epub mobi doc docx odt rtf xls xlsx ods ppt pptx odp
zip rar 7z gz bz2 xz tar tgz zst torrent exe dll msi apk dmg iso bin swf
woff woff2 ttf otf eot
""".split())
NON_PAGE_SUFFIXES = ATTACHMENT_SUFFIXES | {"css", "js", "mjs", "map", "json", "xml", "rss", "atom", "txt", "csv"}
ATTACHMENT_PATH = re.compile(r"/(?:files?|uploads?|attachments?|thumbs?|thumbnails?|images?|audio|video|media|src)(?:/|\.|$)", re.I)
FILE_NAMESPACES = ("file:", "image:", "media:", "special:filepath", "special:redirect/file")


def decoded(value):
    for _ in range(2):
        value = unquote(value)
    return value.casefold()


def has_suffix(value, suffixes):
    return any(match.group(1) in suffixes for match in re.finditer(r"\.([a-z0-9]+)(?=$|[/?;#])", decoded(value)))


def attachment_url(url):
    try:
        parsed = urlsplit(url)
        path = decoded(parsed.path)
        query = parse_qsl(parsed.query, keep_blank_values=True)
    except ValueError:
        return True
    if parsed.hostname in WIKIPEDIA_HOSTS and wikipedia_article(url) is not None:
        return False  # /wiki/Audio or /wiki/Image is an HTML article, not a media path.
    if ATTACHMENT_PATH.search(path) or has_suffix(path, ATTACHMENT_SUFFIXES):
        return True
    if any(segment.startswith(FILE_NAMESPACES) for segment in path.split("/")) or "/special:redirect/file/" in path:
        return True
    for key, value in query:
        key, value = decoded(key), decoded(value)
        if key in {"download", "attachment", "filename", "file", "media"}:
            return True
        if has_suffix(value, ATTACHMENT_SUFFIXES) or (key == "title" and value.startswith(FILE_NAMESPACES)):
            return True
    return False


def page_url(url):
    """Whether a URL can identify an HTML article/board/thread, before fetching."""
    try:
        if urlsplit(url).hostname in WIKIPEDIA_HOSTS:
            return wikipedia_article(url) is not None
    except ValueError:
        return False
    if attachment_url(url):
        return False
    try:
        parsed = urlsplit(url)
        if has_suffix(parsed.path, NON_PAGE_SUFFIXES):
            return False
        for key, value in parse_qsl(parsed.query, keep_blank_values=True):
            if decoded(key) == "action" and decoded(value) not in {"", "view"}:
                return False
            if decoded(key) in {"format", "ctype"} and decoded(value) not in {"", "html"}:
                return False
    except ValueError:
        return False
    return True


def html_headers(headers):
    mime = headers.get("content-type", "").split(";", 1)[0].strip().lower()
    disposition = headers.get("content-disposition", "").split(";", 1)[0].strip().lower()
    return mime in {"text/html", "application/xhtml+xml"} and disposition != "attachment"
