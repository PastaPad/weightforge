"""Remove link addresses from written text without discarding ordinary link labels."""
import re

TEXT_POLICY_VERSION = 'user-text-no-links-v1'

# Schemes, www addresses, bare domains (including internationalized labels),
# mail links, and post-reference links. Never run this on crawl/provenance URLs.
LABEL = r'[^\W_](?:[\w-]{0,61}[^\W_])?'
DOMAIN = rf'(?:{LABEL}\.)+(?:[^\W\d_]{{2,63}}|xn--[a-z0-9-]+)'
ADDRESS = re.compile(
    rf'(?<![\w@])(?:[a-z][a-z0-9+.-]*://[^\s<>"`]+|mailto:[^\s<>"`]+|'
    rf'(?:[\w.+-]+@)?{DOMAIN}(?::\d{{1,5}})?(?:[/\?#][^\s<>"`]*)?)', re.I)
MARKDOWN_LINK = re.compile(r'!?\[([^\]\n]*)\]\((?:https?://|ftp://|www\.)[^\s<>]+\)', re.I)
POST_LINK = re.compile(r'(?<!\w)>{2,3}(?:/[a-z0-9_-]+/)?\d+\b|(?<!\w)>>>/[a-z0-9_-]+/', re.I)


def strip_links(text):
    def punctuation(match):
        address, suffix = match.group(), ''
        pairs = {')': '(', ']': '[', '}': '{'}
        while address and address[-1] in '.,;:!?)]}':
            end = address[-1]
            if end in pairs and address.count(end) <= address.count(pairs[end]):
                break
            address, suffix = address[:-1], end + suffix
        return suffix

    text = MARKDOWN_LINK.sub(lambda match: '' if match.group().startswith('!') else match[1], text)
    text = ADDRESS.sub(punctuation, text)
    text = POST_LINK.sub('', text)
    # Keep paragraph/newline structure, but close gaps left by removed addresses.
    text = re.sub(r'[^\S\n]+', ' ', text)
    text = re.sub(r' +([,.;:!?])', r'\1', text)
    text = re.sub(r'\(\s*\)|\[\s*\]|\{\s*\}', '', text)
    return '\n'.join(line.strip() for line in text.splitlines()).strip()
