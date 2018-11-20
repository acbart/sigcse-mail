import os, sys, requests, json, bs4, time, re
from collections import Counter
from bs4 import Comment
from pprint import pprint
from tqdm import tqdm
import codecs
import html2text
#from BeautifulSoup import BeautifulSoup, SoupStrainer
from urllib.parse import urlparse, parse_qs
from unidecode import unidecode
#from urlparse import urlparse, parse_qs
import random
random.seed(100)

# Is binary?
textchars = bytearray({7,8,9,10,12,13,27} | set(range(0x20, 0x100)) - {0x7f})
def is_binary_string(byts):
    return bool(byts.translate(None, textchars))

# HTTP Access
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36'}
LIST_SERV_PREFIX = 'https://listserv.acm.org'

def force_get(url, path=None):
    if os.path.exists(path):
        with open(path, 'rb') as local_file:
            return local_file.read()
    success = False
    while not success:
        try:
            content = requests.get(url, headers=headers).content
            success = True
        except requests.exceptions.ConnectionError as e:
            print("Connection Error", str(e))
            time.sleep(3)
    with open(path, 'wb') as out:
        out.write(content)
    return content

def build_index():
    print("Building index")
    MAIN = '/SCRIPTS/WA-ACMLPX.CGI?A0=SIGCSE-members'
    PERIODICAL_PATTERN = '/SCRIPTS/WA-ACMLPX.CGI?A1='
    index_page = force_get(LIST_SERV_PREFIX+MAIN, 'indexes.html')
    soup = bs4.BeautifulSoup(index_page, 'lxml')
    anchors = soup.find_all('a')
    with open('indexes.txt', 'w') as indexes:
        for anchor in anchors:
            url = anchor.get('href')
            if url and url.startswith(PERIODICAL_PATTERN):
                indexes.write(LIST_SERV_PREFIX+url+'\n')

def retrieve_periodicals():
    print("Retrieving periodicals")
    URLS = open('indexes.txt', 'r').readlines()
    for id, an_index in enumerate(tqdm(URLS)):
        path = 'periodicals/{}.html'.format(id)
        if os.path.exists(path):
            continue
        force_get(an_index.strip(), path)
        #time.sleep(3)
        
def clean_whitespace(astr):
    return ' '.join(astr.strip().split())

filetypes = Counter()
def process_email(content, resource_id):
    resource_path = os.path.join('parsed_emails', resource_id+'.json')
    soup = bs4.BeautifulSoup(content, 'lxml')
    subject_line = soup.find_all('b', text='Subject:')
    if not subject_line:
        soup = bs4.BeautifulSoup(content, 'html.parser')
        subject_line = soup.find_all('b', text='Subject:')
        if not subject_line:
            raise Exception("Subject line not found! Email ID: "+str(resource_id))
    table = subject_line[0].parent.parent.parent.parent
    tabular_data = [ [clean_whitespace(td.text) for td in tr.find_all('td')]
                     for tr in table.find_all('tr') ]
    try:
        Subject, From, Reply_to, Date, ContentType  = [v[-1] for v in tabular_data[:5]]
    except ValueError:
        print("ERROR")
        print(resource_id)
        print(table)
        return
    attachment_urls = table.find_all('a')
    Attachments = []
    AttachmentTypes = []
    for a in attachment_urls:
        url = a.get('href')
        if not url or not url.startswith('/SCRIPTS/WA-ACMLPX.CGI?A3=ind'):
            continue
        filename, filetype = parse_listserv_url(url, 'A3')
        if filename is None:
            continue
        extension = infer_extension(filetype)
        path = os.path.join('attachments', filename+'.'+extension)
        content = force_get(LIST_SERV_PREFIX+url, path)
        if extension == 'txt' and content:
            pretags= bs4.BeautifulSoup(content, 'lxml').select('pre')
            if pretags:
                content = pretags[0].getText()
                with open(path, 'wb') as out:
                    out.write(content.encode('utf-8'))
        Attachments.append(filename+'.'+extension)
        AttachmentTypes.append(filetype)
        filetypes[filetype] += 1
    email = {
        'subject': Subject,
        'from': From.replace(' <[log in to unmask]>', ''),
        'reply': Reply_to.replace(' <[log in to unmask]>', ''),
        'date': Date,
        'type': ContentType,
        'attachments': Attachments,
        'types': AttachmentTypes
    }
    with open(resource_path, 'w') as output:
        json.dump(email, output)
        
def clean_body(text):
    if not text:
        return ''
    kept_lines = []
    for line in text.split("\n"):
        line = line.replace('[log in to unmask]', '')
        if line == '-- ':
            break
        if line == '--':
            break
        if line == '--------------------------------------':
            break
        if line == '############################':
            break
        if line.startswith('-----Original Message-----'):
            break
        if line.startswith('________________________________'):
            break
        if line.startswith('On ') and line.endswith(' wrote:'):
            break
        if line.startswith('From: '):
            break
        if line.startswith('Sent from my iPhone'):
            break
        if line.startswith('Sent from my BlackBerry'):
            break
        if line.strip() == 'To unsubscribe from the SIGCSE-members list, click the following link:':
            break
        if line.strip().startswith('http://listserv.acm.org/SCRIPTS/WA-ACMLPX.CGI?'):
            continue
        if line.strip() in ('Print', '---|---|---', '* * *', '|  |'):
            continue
        if line.lstrip().startswith('>'):
            continue
        kept_lines.append(line)
    return '\n'.join(kept_lines)
    
def parse_listserv_url(url, id_loc):
    parameters = parse_qs(urlparse(url).query)
    if id_loc not in parameters:
        return None
    resource_id = parameters[id_loc][0]
    filename = parameters['P'][0] + '_' + resource_id
    filetype = parameters.get('T', [None])[0]
    return filename, filetype
        
#/SCRIPTS/WA-ACMLPX.CGI?A2=ind1112E&L=SIGCSE-members&F=&S=&P=82
def retrieve_emails():
    print("Get emails")
    periodicals = os.listdir('periodicals/')
    #random.shuffle(periodicals)
    for page in tqdm(periodicals):
        path = os.path.join('periodicals', page)
        with open(path, 'rb') as inp:
            content = inp.read()
        soup = bs4.BeautifulSoup(content, 'lxml')
        for table_row in soup.findAll('tr', attrs={"class": ["normalgroup", "emphasizedgroup"]}):
            subject, author, date, size = table_row.findAll('p', attrs={"class": "archive"})
            url = subject.span.a['href']
            filename, filetype = parse_listserv_url(url, 'A2')
            path = os.path.join('emails', filename+'.html')
            if not url.startswith(LIST_SERV_PREFIX):
                url = LIST_SERV_PREFIX+url
            content = force_get(url, path)
            process_email(content, filename)
#%%
h2t_handler = html2text.HTML2Text()
h2t_handler.ignore_links = True
h2t_handler.ignore_images = True
def clean_attachments():
    attachments = os.listdir('attachments/')
    #random.shuffle(attachments)
    for attachment in tqdm(attachments):
        path = os.path.join('attachments', attachment)
        with open(path, 'rb') as inp:
            content = unidecode(inp.read().decode('latin1'))
        #if is_binary_string(content):
        #    continue
        #soup = bs4.BeautifulSoup(content, 'html.parser')
        if path.endswith('.html'):
            #soup = bs4.BeautifulSoup(content, 'html.parser')
            #body = soup.find_all('body')
            body = h2t_handler.handle(str(content))
            #for blockquote in body.find_all('blockquote'):
            #    blockquote.decompose()
            #for element in body.find_all(text=lambda text: isinstance(text, Comment)):
            #    element.extract()
            #body = body.get_text()
            body= clean_body(body)
        elif path.endswith('.txt'):
            body= clean_body(content)
        else:
            continue
        #body = soup.find('body')
        #body = clean_body(body)
        cleaned_path = os.path.join('cleaned_attachments', attachment)
        with codecs.open(cleaned_path , 'w', 'utf-8') as out:
            out.write(body)
            #%%
def process_threads():
    periodicals = os.listdir('periodicals/')
    #random.shuffle(periodicals)
    threads = {}
    for page in tqdm(periodicals):
        path = os.path.join('periodicals', page)
        with open(path, 'rb') as inp:
            content = inp.read()
        soup = bs4.BeautifulSoup(content, 'lxml')
        title = soup.select("td.emphasizedgroup h2")[0].text.strip()
        toprow = soup.find("tr", attrs={"class": "emphasizedcell"})
        for subsequent in toprow.find_next_siblings():
            sub_class = subsequent.get("class")
            if not sub_class:
                thread_id = title + ','+ subsequent.find('p').a.attrs['name']
                threads[thread_id] = []
                # New thread
            else:
                # Child thread
                subject, author, date, size = subsequent.findAll('p', attrs={"class": "archive"})
                url = subject.span.a['href']
                filename, filetype = parse_listserv_url(url, 'A2')
                threads[thread_id].append(filename)
    with open('threads.json', 'w') as out:
        json.dump(threads, out, indent=2)
        '''
        for table_row in soup.find_all('tr'):
            if table_row.get('class') in ('normalgroup', "emphasizedgroup"):
                subject, author, date, size = table_row.findAll('p', attrs={"class": "archive"})
                url = subject.span.a['href']
                filename, filetype = parse_listserv_url(url, 'A2')
                path = os.path.join('emails', filename+'.html')
                #print(filename)
            else:
                img = table_row.findAll('img', title="New Thread")
                print(img)
                #url = subject.span.a['href']
        '''
#%%
def infer_extension(filetype):
    filetype = filetype.lower()
    if filetype.startswith('image/'):
        return filetype.split('/')[1]
    elif filetype == 'text/html':
        return 'html'
    elif 'text' in filetype:
        return 'txt'
    elif 'pdf' in filetype:
        return 'pdf'
    elif 'msword' in filetype:
        return 'doc'
    elif 'excel' in filetype:
        return 'xls'
    else:
        return 'unknown'
def test_extensions():
    with open('attachment_type_freqs.json', 'r') as inp:
        freqs = json.load(inp)
    extensions = Counter()
    for filetype, freq in freqs.items():
        extensions[infer_extension(filetype)] += freq
    pprint(dict(extensions.items()))
#test_extensions()
#retrieve_emails()
#%%
#build_index()
retrieve_periodicals()
#retrieve_emails()
#pprint(dict(filetypes.items()))
#%%
#clean_attachments()
#process_threads()