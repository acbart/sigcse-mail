# sigcse-mail
A quantitative analysis of the SIGCSE-members listserv

# Orientation

Hi! I'm glad you're interested in my analysis. This is a little cluttered, but I've tried to establish some organization. The best stuff is in `sigcse-mail_main-analysis.ipynb`. If you're interested in running my analysis on your own, you'll want to jump down to [Full Process](#Full Process).

# Main Analysis

If you check out [sigcse-mail_main-analysis.ipynb](sigcse-mail_main-analysis.ipynb) then you can see the fruits of my labor as a Jupyter Notebook. Note that without obtaining a copy of `data/sigcse-emails.db` from [me](#Getting the Dataset) or regenerating it yourself, you won't be able to run this notebook.

For my qualitative analysis, check out [results/LongThreadAnalysis.xlsx](results/LongThreadAnalysis.xlsx) or preview the [HTML version](http://htmlpreview.github.io/?https://github.com/acbart/sigcse-mail/blob/master/results/long_thread_analysis.html). The current organization is basically a series of rows where:

* Thread title
* Year/month
* My stream-of-consciousness summary
* Quote URL
* Interesting quote from thread

Note that the first three columns are merged vertically across all the quotes in the thread. It's pretty hard to read, so I need to mess with this.

# Full Process

First, you will need to download your own copy of the listserv using the functions in `sigcseArchive.py`. 

    > from sigcseArchive import *
    > build_index()
    > retrieve_periodicals()
    > retrieve_emails()
    > clean_attachments()
    > process_threads()

This is a rough outline of them, and some may require multiple attempts as you sort out unicode issues and the like. Further, I think the system has changed how it does verification. You might need to fight with it a little.

After you have all the data downloaded, I used `dump_to_sqlite.py` to generate a single SQL database that would be used by the analysis scripts. From here, you should be able to go do the [Main Analysis](#Main Analysis).

# Website

You'll notice there's a website folder. I was working on a cool web interface, before I was told to not share the dataset. I still used it internally during my qualitative analysis to read threads more easily. It's a little messy - this was a very ad-hoc project! No server is required to run them, and you can easily move the generated folder around between filesystems (I read the longer threads on a tablet).

# Getting the Dataset

I'm afraid I've been asked to not publicly provide the emails. There are concerns that members may feel their privacy was violated. In order to share the dataset with you, you must prove that you're an ACM member. I don't really have a way of validating ACM member numbers, so I can't really do this very effectively myself. In other words, unless you show up in the dataset I probably can't share it with you. But if you do, and you want it for your own version of your analysis, I'm happy to share it. Just email me at acbart@udel.edu and establish your credentials in a reasonable way (e.g., use your university email, not your hotmail account).

# Improving My Analysis

As I mentioned at my SIGCSE talk, I'd love to see someone improve my analysis. I think there's a cool project here involving natural language processing, or some great history for someone who's better at qualitative analysis. If you end up doing anything, please let me know. I'm happy to accept good pull requests.

Here are some targets I think might be interesting:

* Thread cleaning: My thread cleaning method was pretty shaky, and often too aggressive. It turns out that email cleaning is a hard problem and there's only so much research on it. Members often quote text in weird ways, and sometimes they do things like interleave quotes and new stuff. You could write a much more complex cleaning function that does a better job filtering out white noise.
* Better automatic summarization of email themes: I read over several papers on email summarizations, but didn't find any techniques that I felt I could apply easily enough. But I think there's some cool stuff with multiple-document summarization that might apply. It'd be great to get a more serious look at that!
* Better thread detection: currently, I relied on the listserv's concept of threads, but during my qualitative analysis it became obvious that their model is limited. First of all, they fail to merge threads across week boundaries! Second, they are fooled by someone minutely changing the title of a thread. There's probably some cool things to do with edit distance, temporal distance, and quoted material to figure out if two threads should be joined. There's also probably a cool diagram involving forking that you could make.
