import re

webenc = 'utf-8'
skeleton = 'www/skeleton.html'
skeleton_indent = '    '


default_w = 8
default_p = 24
default_s = 'pi'

def format_webform(form, w, p, s):
    return form.format(str(s), str(w), str(p))

assets = {
    '$form$' : ('www/form.html',
                lambda x : format_webform(x, default_w, default_p, default_s),),
}
pages = {
    'index' : ('www/index.html', 'text/html',),
    'about' : ('www/about.html', 'text/html',),
    'favicon.ico' : ('www/favicon.ico', 'image/x-icon',),
}
root_page = 'index'
web_form = '$form$'

protocols = {'demo', 'json', 'text' 'fmt', 'jfmt'}

with open(skeleton, encoding=webenc, mode='r') as f:
    skeleton_content = f.read()

def pre(s):
    return '<pre>\n' + s.strip() + '\n</pre>'

def skeletonize(s, indent=False):
    s = s.strip()
    if indent:
        s = skeleton_indent + s.replace('\n', '\n' + skeleton_indent)
    return bytes(skeleton_content.format(s), webenc)

# custom, add-hoc html rewriting

re_assets = re.compile(r'|'.join(re.escape(k) for k in assets.keys()))
re_indent = re.compile(r'^(\s+)\Z', flags=re.M)

def import_asset(path, formatter):
    with open(path, encoding=webenc, mode='r') as f:
        s = f.read()
        return s.strip(), formatter

asset_content = {name : import_asset(path, formatter) for name, (path, formatter,) in assets.items()}

def process_assets(s):
    to_replace = re_assets.findall(s)
    finished = ''
    for k in to_replace:
        left, _, right = s.partition(k)
        indent_by = re_indent.search(left)
        asset, formatter = asset_content[k]
        if indent_by is None:
            asset = formatter(asset)
        else:
            asset = formatter(asset).replace('\n', '\n' + indent_by.group(1))
        finished += left + asset
        s = right
    return finished + s

def import_page(path):
    if path.endswith('.html'):
        with open(path, encoding=webenc, mode='r') as f:
            s = f.read()
        s = process_assets(s)
        return skeletonize(s, indent=True)
    else:
        with open(path, mode='rb') as f:
            data = f.read()
        return data

def re_prefix(keys):
    return re.compile(r'/(' + r'|'.join(re.escape(k) for k in keys) + r')(\.html?)?/*')

page_re = re_prefix(pages.keys())
empty_re = re.compile(r'/*')
protocol_re = re_prefix(protocols)
page_content = {name : (import_page(path), ctype,) for name, (path, ctype,) in pages.items()}
