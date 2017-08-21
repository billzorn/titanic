import re

webenc = 'utf-8'
skeleton = 'www/skeleton.html'
skeleton_indent = '    '

pages = {
    'index' : ('www/index.html', 'text/html',),
    'about' : ('www/about.html', 'text/html',),
    'favicon.ico' : ('www/favicon.ico', 'image/x-icon',),
}
root_page = 'index'

protocols = {'demo', 'json', 'text'}

with open(skeleton, encoding=webenc, mode='r') as f:
    skeleton_content = f.read()

def pre(s):
    return '<pre>\n' + s.strip() + '\n</pre>'

def skeletonize(s, indent=False):
    s = s.strip()
    if indent:
        s = skeleton_indent + s.replace('\n', '\n' + skeleton_indent)
    return bytes(skeleton_content.format(s), webenc)

def import_page(path):
    if path.endswith('.html'):
        with open(path, encoding=webenc, mode='r') as f:
            s = f.read()
            return skeletonize(s, indent=True)
    else:
        with open(path, mode='rb') as f:
            data = f.read()
        return data

def re_prefix(keys):
    return re.compile(r'/(' + r'|'.join(re.escape(k) for k in keys) + r')(\.html?)?/*')

page_re = re_prefix(pages.keys())
protocol_re = re_prefix(protocols)
page_content = {name : (import_page(path), ctype,) for name, (path, ctype,) in pages.items()}
