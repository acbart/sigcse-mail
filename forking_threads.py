'''
I attempted to do some analysis where I could join up forked threads.

For example, when someone spawns a new thread ("Re: <Old thing>"), it would be
nice to include that in the original thread's count. But that appears to be
pretty tricky.
'''

# But after I ran the below code, I realized that I'm not really missing that
# much with these threads.
emails.groupby(lambda x: emails.subject.loc[x] 
               if not emails.subject.loc[x].startswith("Re:") 
               else emails.subject.loc[x][4:]).kind.count().plot.hist()
emails.groupby('subject').kind.count().plot.hist()
plt.show()

# Do threads fork a lot?
#editdistance.eval
#pd.merge(emails.subject, emails.subject)
thread_starters = emails.query('kind == "Normal"').groupby('thread_id').head(1)
#emails[unique_thread_ids]
import editdistance
#thread_starters.subject.combine(thread_starters.subject, editdistance.eval)
for thread in islice(thread_starters.itertuples(), 0):
    months_before = thread.sent - pd.Timedelta(days=60)
    months_after = thread.sent + pd.Timedelta(days=60)
    four_month_range = ((months_before < thread_starters.sent) & 
                        (thread_starters.sent < months_after))
    subjects = thread_starters[four_month_range].drop(thread.Index).subject
    ds = subjects.apply(lambda s, s2: editdistance.eval(s, s2), 
                                                         s2=thread.subject)
    if (len(ds[ds == 0])) > 0:
        print(thread)
    #print(thread_starters[thread_starters.sent < thread.sent])
#pd.Series(list(combinations(thread_starters.subject, 2))).apply(lambda x: editdistance.eval(*x))
#pd.merge(thread_starters.subject, thread_starters.subject, on='index')
forks = thread_starters[thread_starters.subject.str.startswith("Re:")]
#threads.loc[thread_starters[thread_starters.subject.isin(forks.subject.str[4:])].thread_id]
#thread_starters[thread_starters.subject.isin(forks.subject.str[4:])]
forks.subject.str[4:]