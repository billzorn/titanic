import re
import urllib

# defaults

webenc = 'utf-8'
skeleton = 'www/skeleton.html'
skeleton_indent = '    '

maxlen = 1536
default_s = 'pi'
maxw = 20
default_w = 8
maxp = 1024
default_p = 24

def format_webform(form, s = default_s, w = default_w, p = default_p):
    return form.format(s=s, w=w, p=p,
                       maxlen=maxlen, maxw=maxw, maxp=maxp)

# content directory listing

root_page = 'index'
web_form = '$form$'
err_body = '$err$'

assets = {
    web_form : ('www/form.html', format_webform,),
    err_body : ('www/error.html', None,),
}
pages = {
    root_page : ('www/index.html', 'text/html',),
    'about'   : ('www/about.html', 'text/html',),
    'favicon.ico'  : ('www/favicon.ico', 'image/x-icon',),
    'piceberg.png' : ('www/piceberg.png', 'image/png',),
    'reals.pdf' : ('www/reals.pdf', 'application/pdf',),
    'ulps.pdf'  : ('www/ulps.pdf', 'application/pdf',),
}

protocols = {'demo', 'fmt'}

with open(skeleton, encoding=webenc, mode='r') as f:
    skeleton_content = f.read().strip() + '\n'

# html formatting helpers

def link(s, s_link, w, p):
    href = '/demo?s={}&w={:d}&p={:d}'.format(urllib.quote_plus(s_link), w, p)
    return '<a href="{}">{}</a>'.format(href, s)

re_indent_match = re.compile(r'(\s*\n)([^\n])')
def indent(s, indent_by):
    if len(s) > 0:
        replace = r'\1' + indent_by + r'\2'
        return indent_by + re_indent_match.sub(replace, s)
    else:
        return s

def pre(s, nl=True):
    if nl:
        return '<pre>\n' + s.rstrip() + '\n</pre>'
    else:
        return '<pre>' + s.rstrip() + '\n</pre>'

def skeletonize(s, ind=False):
    s = s.strip()
    if ind:
        s = indent(s, skeleton_indent)
    return skeleton_content.format(s)

def webencode(s):
    return bytes(s, webenc)

# shared assets

cre_assets = r'|'.join(re.escape(k) for k in assets.keys())
re_split_assets = re.compile(r'(.*)(' + cre_assets + r')',
                             flags=re.MULTILINE|re.DOTALL)
re_indent_assets = re.compile(r'^([^\S\n]*)\Z',
                              flags=re.MULTILINE|re.DOTALL)

def import_asset(path, formatter):
    with open(path, encoding=webenc, mode='rt') as f:
        s = f.read()
    return s.strip(), formatter

asset_content = {name : import_asset(path, formatter) for name, (path, formatter,) in assets.items()}

def create_webform(s, w, p):
    asset, _ = asset_content[web_form]
    return format_webform(asset, s, w, p)

def create_error(err, msg):
    asset, _ = asset_content[err_body]
    return asset.format(err=err, msg=msg)

def protocol_headers_body(s, w, p, content):
    form_body = indent(create_webform(s, w, p), skeleton_indent)
    content_body = pre(content)

    headers = (
        ('Content-Type', 'text/html',),
    )
    body = skeletonize(form_body + '\n\n' + content_body, ind=False)

    return headers, webencode(body)

# custom, ad-hoc html rewriting

def process_assets(s):
    asset_groups = re_split_assets.findall(s)
    segments = []
    last_idx = 0
    for s_pre, name in asset_groups:
        asset_indent = re_indent_assets.search(s_pre)
        asset, formatter = asset_content[name]
        segments.append(s_pre[:asset_indent.start(1)])
        if formatter is None:
            segments.append(indent(asset, asset_indent.group(1)))
        else:
            segments.append(indent(formatter(asset), asset_indent.group(1)))
        last_idx += len(s_pre) + len(name)
    segments.append(s[last_idx:])
    return ''.join(segments)

re_bin_ctypes = re.compile(r'image.*|application/pdf',
                           flags=re.IGNORECASE)
re_proc_ctypes = re.compile(r'text/html',
                            flags=re.IGNORECASE)

def import_page(path, ctype):
    if re_bin_ctypes.fullmatch(ctype):
        with open(path, mode='rb') as f:
            data = f.read()
        return data, ctype
    else:
        with open(path, encoding=webenc, mode='rt') as f:
            s = f.read()
        if re_proc_ctypes.fullmatch(ctype):
            s = process_assets(s)
            s = skeletonize(s, ind=True)
        return webencode(s), ctype

# preloaded static pages

page_content = {name : import_page(path, ctype) for name, (path, ctype,) in pages.items()}

# path recognition regexes

cre_empty = r'/*'
empty_re = re.compile(cre_empty)
page_re = re.compile(cre_empty +
                     r'(' + r'|'.join(re.escape(k) for k in pages.keys()) + r')' +
                     r'([.]html?)?')
protocol_re = re.compile(cre_empty +
                         r'(' + r'|'.join(re.escape(k) for k in protocols) + r')')
