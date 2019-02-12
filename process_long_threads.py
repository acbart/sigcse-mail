import json
from pprint import pprint
from jinja2 import Template

# Load in data
with open('results/long_thread_analysis.json', 'r') as analysis_file:
    reviews = json.load(analysis_file)

# Collapse quotes under threads
current = None
posts = []
for review in reviews:
    if review['Thread Name']:
        current = {
            'Summary': review['Quick Summary'],
            'Title': review['Thread Name'],
            'When': review['Year/Month'],
            'Quotes': []
        }
        posts.append(current)
    if review['Relevant quotes'] or review['Post']:
        quote = {
            'Link': review['Post'],
            'Text': review['Relevant quotes']
        }
        current['Quotes'].append(quote)

HTML_TEMPLATE = """
<html>
<head>
<style>
pre {
 white-space: pre-wrap;       /* css-3 */
 white-space: -moz-pre-wrap;  /* Mozilla, since 1999 */
 white-space: -pre-wrap;      /* Opera 4-6 */
 white-space: -o-pre-wrap;    /* Opera 7 */
 word-wrap: break-word;       /* Internet Explorer 5.5+ */
}
</style>
</head>
<body>
{% for post in posts %}
<div>
    <strong>{{ post.Title }}</strong><br>
    <span>{{ post.When }}</span><br>
    {{ post.Summary }}
    <ul>
    {% for quote in post.Quotes %}
        <li><pre>
        {%- if quote.Link %}<small>{{ quote.Link }}: </small>{% endif -%}
        {{- quote.Text -}}
        </pre></li>
    {% endfor %}
    </ul>
</div>
{% endfor %}
</body>
</html>
"""

with open("results/long_thread_analysis.html", 'w') as out:
    print(Template(HTML_TEMPLATE).render(posts=posts), file=out)
