# sigcse-mail
A quantitative analysis of the SIGCSE-members listserv

# Orientation
Okay, so this archive is a little disorganized. However, it *should* have everything you need to get started.

First, you will need to download your own copy of the listserv using the functions in `sigcseArchive.py`. 

    > from sigcseArchive import *
    > build_index()
    > retrieve_periodicals()
    > retrieve_emails()
    > clean_attachments()
    > process_threads()

Then, you can start rerunning the analyses in `process-2.py` and `just_text.py`. Be careful, some of these are very long running. You should run them in an environment like Spyder or IPython that will directly embed MatPlotLib graphs - otherwise, anticipate like a hundred windows appearing.
