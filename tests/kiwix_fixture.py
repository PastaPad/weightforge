"""Small real ZIM archives for offline importer/UI tests."""
from pathlib import Path

from libzim.writer import Creator, Hint, Item, StringProvider


class Article(Item):
    def __init__(self, path, content, mime='text/html'):
        super().__init__()
        self.path, self.content, self.mime = path, content, mime

    def get_path(self):
        return self.path

    def get_title(self):
        return self.path.replace('_', ' ')

    def get_mimetype(self):
        return self.mime

    def get_contentprovider(self):
        return StringProvider(self.content)

    def get_hints(self):
        return {Hint.FRONT_ARTICLE: int(self.mime == 'text/html')}


def make_archive(path, source='en.wikipedia.org'):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def html(text):
        return f'''<html><head><title>Fixture</title></head><body class="ns-0">
            <header>UI header</header><nav>UI menu</nav><main><div class="mw-parser-output">
            <div class="hatnote">UI notice</div><div class="toc"><ul><li>UI contents</li></ul></div>
            <p>{text} The <a href="Other">written word</a> remains. https://example.org/remove</p>
            <figure><img src="image.png"><figcaption>MEDIA caption</figcaption></figure>
            <div class="infobox"><p>UI infobox</p></div><video>MEDIA player</video>
            <h2>References</h2><ul><li>REF bibliography</li></ul><h2>Further discussion</h2><p>Article conclusion.</p>
            </div></main><footer>UI footer</footer></body></html>'''
    with Creator(path).config_indexing(False, 'eng').config_nbworkers(1) as creator:
        for key, value in dict(Title='Offline fixture', Creator='Wikipedia', Publisher='openZIM',
                               Language='eng', Date='2026-09-14', Source=source).items():
            creator.add_metadata(key, value)
        for index in range(8):
            text = f'Topic {index} explains a distinct subject. ' + 'Human-written article prose provides context and detail. ' * 8
            creator.add_item(Article(f'Topic_{index}', html(text)))
            if index == 0:
                creator.add_item(Article('Duplicate', html(text)))
        creator.add_item(Article('Main_Page', html('HOME text')))
        creator.add_item(Article('Category:Topics', html('CATEGORY text')))
        creator.add_item(Article('File:Example', html('FILE text')))
        creator.add_item(Article('Talk:Topic_0', html('TALK text')))
        creator.add_item(Article('index', html('LANDING text')))
        creator.add_item(Article('Unknown_layout', '<p>UNRECOGNIZED text</p>'))
        creator.add_item(Article('Short', '<div class="mw-parser-output"><p>Too short.</p></div>'))
        creator.add_item(Article('image.png', b'MEDIA bytes', 'image/png'))
        creator.add_redirection('Redirect', 'Redirect', 'Topic_0', {Hint.FRONT_ARTICLE: 1})
        creator.add_item(Article('Fragment_redirect', '<html><head><meta content="0;URL=Topic_0#Section" HTTP-EQUIV="Refresh"></head><body><a href="Topic_0#Section">Redirect</a></body></html>'))
        creator.set_mainpath('index')
    return path
