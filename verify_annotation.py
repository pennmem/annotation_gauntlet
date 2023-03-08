#!python
import sys
import re
import glob
import os
import textwrap
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

ACCEPTABLE_TIME_DIFF=25

PASS_IS_VOCALIZATION=False

EXCLUDE_MATCHES_GREATER_THAN=1000

PLOT_ABSOLUTE_VALUE=True

class ComparisonTypes:
    MATCH=0
    MISMATCH=1
    QC_ONLY=2
    TEST_ONLY=3

class Comparison:

    def __init__(self, type, QC_word=None, test_word=None, 
                             QC_time=None, test_time=None,
                             explanation=''):
        (self.type, self.QC_word, self.test_word, self.QC_time, self.test_time, self.explanation) = \
                (type, QC_word, test_word, QC_time, test_time, explanation)
        self.is_QC_voc = QC_word=='<>'
        self.is_test_voc = test_word=='<>'

    @property
    def diff(self):
        if self.type==ComparisonTypes.MATCH or self.type==ComparisonTypes.MISMATCH:
            return self.QC_time - self.test_time
        else:
            return None

    @property
    def time(self):
        if self.type==ComparisonTypes.TEST_ONLY:
            return self.test_time
        else:
            return self.QC_time
    
    @property
    def word(self):
        if self.type==ComparisonTypes.MISMATCH:
            return '%s/%s'%(self.QC_word, self.test_word)
        if self.type==ComparisonTypes.QC_ONLY or self.type==ComparisonTypes.MATCH:
            return self.QC_word
        else:
            return self.test_word

    @property
    def label(self):
        l = '%.0f\t%s\t%s'%(self.time, self.word, '(%.1f)'%self.diff if self.diff else '')
        if self.explanation:
            e = '\n'+'\t'*3 + ('\n'+'\t'*3).join(textwrap.wrap(self.explanation, width=60))
        else:
            e = ''
        return l+e

    @property
    def in_QC(self):
        return self.type!=ComparisonTypes.TEST_ONLY
   
class ComparisonContainer(object):

    def __init__(self, items=None):
        self.items = (items if items!=None else [])
        pass

    def __iter__(self):
        for item in self.items:
            yield item

    def __len__(self):
        return len(self.items)

    def append(self, item):
        self.items.append(item)
    
    def extend(self, comparisons):
        self.items.extend(comparisons.items)

    def filter(self, rule):
        newContainer = ComparisonContainer()
        for item in self:
            if rule(item):
                newContainer.append(item)
        return newContainer

    @property
    def label(self):
        return ('\n'.join(['\t\t%s'%item.label for item in self.items]))

    def flatten(self):
        container = ComparisonContainer()
        for innerComp in self:
            container.extend(innerComp)
        return container

def check_inputs():
    if len(sys.argv)<2:
        print('Usage: python check_annotation.py </path/to/sess/to/check> </path/to/gold/standard> --hist')
        exit()

def get_path_part(session_dir, directories_up):
    abspath = os.path.abspath(session_dir)
    split_path = abspath.split('/')
    return split_path[-directories_up]

def get_session(session_dir):
    split_sess_dir = session_dir.split('_')
    if len(split_sess_dir) > 1:
        return split_sess_dir[1]
    else:
        return '????????'

def get_experiment(session_dir):
    try:
        return get_path_part(session_dir, 2)
    except:
        return '??????'

def get_subject(session_dir):
    try:
        return get_path_part(session_dir, 4)
    except:
        return '??????'


def get_annotator(file_to_parse):
    lines = [x.strip() for x in open(file_to_parse).readlines()]
    for line in lines:
        if re.match('#Annotator:.*', line):
            return line[11:]

def is_annotation_line(line):
    if re.match('[0-9]+\.[0-9]+\s+-?[0-9]+\s+(([A-Z]+)|(<>))\s*(\[.*\])?', line):
        return True
    else:
        return False


def get_annotation(line):
    split_line = line.split('\t')
    time = float(line.split()[0])
    word = line.split()[2]
    if len(line.split())>3:
        explanation = ' '.join(line.split()[3:])
    else:
        explanation = ''
    if PASS_IS_VOCALIZATION and word=='PASS':
        word = '<>'
    return (time, word, explanation)


def get_annotation_list(file_to_parse):
    explanation_file = file_to_parse+'.explanation'
    try:
        lines = [x.strip() for x in open(explanation_file).readlines()]
    except:
        lines = [x.strip() for x in open(file_to_parse).readlines()]
    annotations = []
    for line in lines:
        if is_annotation_line(line):
            annotations.append(get_annotation(line))
    return annotations


def get_times_and_words(annotation):
    times = [x[0] for x in annotation]
    words = [x[1] for x in annotation]
    explanations = [x[2] for x in annotation]
    return times, words, explanations


def get_matching_indices(word_list, word_to_match):
    matches = []
    start = 0
    try:
        while True: # Loop will exit on exception 
            index = word_list.index(word_to_match, start)
            matches.append(index)
            start = index+1
    except:
        pass
    return matches


def get_closest_time(testing_times, true_time):
    """
    returns: (best_diff, index_of_best_diff)
    """
    time_differences  = []
    diffs = [true_time-testing_time for testing_time in testing_times]
    abs_diffs = [abs(diff) for diff in diffs]
    min_diff = min(abs_diffs)
    min_index = abs_diffs.index(min_diff)
    return testing_times[min_index], min_index


def compare_annotation_lists(testing_annotations, true_annotations):
    testing_annotations_used = [False]*len(testing_annotations)
    testing_times, testing_words, _ = get_times_and_words(testing_annotations)
    true_times, QC_words, explanations = get_times_and_words(true_annotations)
    results = ComparisonContainer()
    
    for QC_word, true_time, explanation in zip(QC_words, true_times, explanations):
        testing_indices = get_matching_indices(testing_words, QC_word)
        if len(testing_indices)==0:
            result, best_index = make_nonmatch(testing_times, true_time, QC_word, testing_words, explanation)
            results.append(result)
            if best_index:
                testing_annotations_used[best_index] = True
            continue
        this_testing_times = [testing_times[i] for i in testing_indices]
        closest_time, ind = get_closest_time(this_testing_times, true_time)
        if abs(closest_time-true_time) > EXCLUDE_MATCHES_GREATER_THAN:
            result, _ = make_nonmatch(testing_times, true_time, QC_word, testing_words, explanation)
            results.append(result)
            continue
        if not testing_annotations_used[testing_indices[ind]]:
            testing_annotations_used[testing_indices[ind]] = True
            results.append(Comparison(ComparisonTypes.MATCH,
                                      QC_word=QC_word,
                                      QC_time=true_time,
                                      test_time=closest_time,
                                      explanation=explanation))
        else:
            results.append(Comparison(ComparisonTypes.QC_ONLY,
                                      QC_word=QC_word,
                                      QC_time = true_time,
                                      explanation=explanation))

    for used, word, time in zip(testing_annotations_used, testing_words, testing_times):
        if not used:
            results.append(Comparison(ComparisonTypes.TEST_ONLY,
                                      test_word=word,
                                      test_time=time))

    return results


def make_nonmatch(testing_times, true_time, QC_word, testing_words, explanation):
    closest_time, best_index = get_closest_time(testing_times, true_time)
    diff = closest_time - true_time
    if abs(diff) < ACCEPTABLE_TIME_DIFF:
        result = Comparison(ComparisonTypes.MISMATCH,
                                  QC_word=QC_word,
                                  QC_time=true_time,
                                  test_word=testing_words[best_index],
                                  test_time=closest_time,
                                  explanation=explanation)
    else:
        result = Comparison(ComparisonTypes.QC_ONLY,
                                  QC_word=QC_word, 
                                  QC_time=true_time,
                                  explanation=explanation)
        best_index = None
    return result, best_index



def get_num_from_file(filename):
    basename = os.path.basename(filename)
    no_ext = os.path.splitext(basename)[0]
    num = str.replace(no_ext, '_', '.')
    try:
        return float(num)
    except:
        return -1


def file_cmp(f1, f2):
    n1 = get_num_from_file(f1)
    n2 = get_num_from_file(f2) 
    return cmp(n1,n2)


def get_annotation_files(folder):
    files = glob.glob(os.path.join(folder, '*.ann'))
    if len(files)==0:
        raise Exception('No .ann files in %s' % folder)
    files.sort(cmp=file_cmp) 
    return files

def summarize_all_comparisons(comparisons, test_files, total):
    print_summary(comparisons.flatten(), False, True)
    for file, comparison in zip(test_files, comparisons):
        print_summary(comparison, True, False, os.path.basename(file))

def print_summary(comparisons, show_details, show_totals, label=None):
    words_in_QC = comparisons.filter(lambda c: c.in_QC and not c.is_QC_voc)
    vocs_in_QC = comparisons.filter(lambda c: c.in_QC and c.is_QC_voc)
    
    words_only_in_QC = words_in_QC.filter(lambda c: c.type==ComparisonTypes.QC_ONLY)
    words_only_in_test = comparisons.filter(lambda c: c.type==ComparisonTypes.TEST_ONLY and not c.is_test_voc)

    vocs_only_in_QC = comparisons.filter(lambda c: c.type==ComparisonTypes.QC_ONLY and c.is_QC_voc)
    vocs_only_in_test = comparisons.filter(lambda c: c.type==ComparisonTypes.TEST_ONLY and c.is_test_voc)

    mismatches = comparisons.filter(lambda c: c.type==ComparisonTypes.MISMATCH)
    word_mismatches = mismatches.filter(lambda c: not c.is_QC_voc)
    voc_mismatches = mismatches.filter(lambda c: c.is_QC_voc)
    
    timing_errors = comparisons.filter(lambda c:c.type==ComparisonTypes.MATCH and abs(c.diff)>ACCEPTABLE_TIME_DIFF)
    timing_vocs = timing_errors.filter(lambda c:c.is_QC_voc)
    timing_words = timing_errors.filter(lambda c: not c.is_QC_voc)
    
    perc = lambda a,b: 100*float(len(a))/len(b)

    if show_totals:
        output = """
Total words in QC: %(num_words)d
Total vocalizations in QC %(num_vocs)d

Words only in QC: %(num_QC_words)d (%(perc_QC_words).2f%%)
Words only in test: %(num_test_words)d (%(perc_test_words).2f%%)

Vocalizations only in QC: %(num_QC_vocs)d (%(perc_QC_vocs).2f%%)
Vocalizations only in test: %(num_test_vocs)d (%(perc_test_vocs).2f%%)

Mismatches on words: %(num_mism)d (%(perc_mism).2f%%)
Mismatches on vocalizations: %(num_mism_vocs)d (%(perc_mism_vocs).2f%%)

Timing errors (words): %(num_time_words)d (%(perc_time_words).2f%%)
Timing errors (vocs): %(num_time_vocs)d (%(perc_time_vocs).2f%%)""" %\
        {'num_words': len(words_in_QC),
         'num_vocs': len(vocs_in_QC),
         'num_QC_words': len(words_only_in_QC),
         'perc_QC_words': perc(words_only_in_QC, words_in_QC),
         'num_test_words': len(words_only_in_test),
         'perc_test_words': perc(words_only_in_test, words_in_QC),
         'num_QC_vocs': len(vocs_only_in_QC),
         'perc_QC_vocs': perc(vocs_only_in_QC, vocs_in_QC),
         'num_test_vocs': len(vocs_only_in_test),
         'perc_test_vocs': perc(vocs_only_in_test, vocs_in_QC),
         'num_mism': len(word_mismatches),
         'perc_mism': perc(word_mismatches, words_in_QC),
         'num_mism_vocs': len(voc_mismatches),
         'perc_mism_vocs': perc(voc_mismatches, vocs_in_QC),
         'num_time_words': len(timing_words),
         'perc_time_words': perc(timing_words, words_in_QC),
         'num_time_vocs': len(timing_vocs),
         'perc_time_vocs': perc(timing_vocs, vocs_in_QC)}
        print output
    
    if show_details:
        output = ""
        if len(words_only_in_QC) != 0:
            output += '\n\tWords only in QC:\n%s'%(words_only_in_QC.label)
        if len(words_only_in_test) != 0:
            output += '\n\tWords only in test:\n%s'%(words_only_in_test.label)
        if len(vocs_only_in_QC) != 0:
            output += '\n\tVocs only in QC:\n%s'%(vocs_only_in_QC.label)
        if len(vocs_only_in_test) != 0:
            output += '\n\tVocs only in test:\n%s'%(vocs_only_in_test.label)
        if len(word_mismatches) != 0:
            output += '\n\tWord mismatches:\n%s'%(word_mismatches.label)
        if len(voc_mismatches) != 0:
            output += '\n\tVoc mismatches:\n%s'%(voc_mismatches.label)
        if len(timing_words) != 0:
            output += '\n\tTiming errors on words:\n%s'%(timing_words.label)
        if len(timing_vocs) != 0:
            output += '\n\tTiming errors on vocs:\n%s'%(timing_vocs.label)
        if len(output) != 0:
            output = '\n%s'%label + output
            print output


def make_histogram(comparisons, is_voc):
    matches = comparisons.filter(lambda x: x.type==ComparisonTypes.MATCH or x.type==ComparisonTypes.MISMATCH)
    isvoc_matches = matches.filter(lambda x: x.is_QC_voc == is_voc)
    isvoc_diffs = [c.diff for c in isvoc_matches]
    if PLOT_ABSOLUTE_VALUE:
        isvoc_diffs = [abs(d) for d in isvoc_diffs]
    mean = np.mean(isvoc_diffs)
    std = np.std(isvoc_diffs)
    plt.hist(isvoc_diffs, 50, facecolor='green', alpha=.75)
    plt.xlabel('Time difference (ms)')
    plt.ylabel('Counts')
    plt.grid(True)
    plt.title(r'$\mathrm{%s}: \mu=%.1fms,\ \sigma=%.1fms$'%('Vocalizations' if is_voc else 'Words', mean, std))
    plt.show()

def compare_session_folders(test_folder, true_folder):
    test_files = get_annotation_files(test_folder)
    true_files = get_annotation_files(true_folder)
    if not (len(test_files) == len(true_files)):
        raise Exception('Number of annotation files in %s and %s do not match'%
                (test_folder, true_folder))
        
    comparisons = ComparisonContainer()
    total_num_anns = 0
    for (test_file, true_file) in zip(test_files, true_files):
        test_ann = get_annotation_list(test_file)
        true_ann = get_annotation_list(true_file)
        total_num_anns += len(true_ann)
        comparison = compare_annotation_lists(test_ann, true_ann)
        comparisons.append(comparison)
   
    test_annotator = get_annotator(test_files[0])
    true_annotator = get_annotator(true_files[0])

    print('%s\nOriginal annotator: %s'%('*'*60, test_annotator))
    print('QC annotator: %s'%true_annotator)
    print('Timing error range: +/-%d ms\n'%ACCEPTABLE_TIME_DIFF)

    print('Subject: %s\nExperiment: %s\nSession: %s'%\
            (get_subject(test_folder),
             get_experiment(test_folder),
             get_session(test_folder)))

    summarize_all_comparisons(comparisons, test_files, total_num_anns)

    return comparisons

def get_session_list(list_file):
    if os.path.isdir(list_file):
        return [list_file]
    return [x.strip() for x in open(list_file).readlines()]

def should_show_hist():
    return '--hist' in sys.argv

def compare_all_sessions(test_file, verification_file):
    test_sessions = get_session_list(test_file)
    verification_sessions = get_session_list(verification_file)
    if len(test_sessions) != len(verification_sessions):
        raise Exception('unequal number of sessions in %s and %s'%
                (test_file, verification_file))
    comparisons = ComparisonContainer()
    for (test_session, ver_session) in zip(test_sessions, verification_sessions):
        comparisons.extend(compare_session_folders(test_session, ver_session))

    if should_show_hist():
        make_histogram(comparisons.flatten(), False)
        make_histogram(comparisons.flatten(), True)



if __name__ == '__main__':
    check_inputs()
    compare_all_sessions(sys.argv[1], sys.argv[2])
