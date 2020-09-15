import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    prob_dist = dict()
    
    if len(corpus[page]) == 0:
        prob = 1 / len(corpus)

        for dest_page in corpus:
            prob_dist[dest_page] = prob

    else:
        prob_no_link = (1 - damping_factor) / len(corpus)
        prob_has_link = damping_factor / len(corpus[page]) + prob_no_link

        for dest_page in corpus:
            if dest_page in corpus[page]:
                prob_dist[dest_page] = prob_has_link
            else:
                prob_dist[dest_page] = prob_no_link

    return prob_dist


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    
    # Initialize dictionary.
    PR = dict()
    for page in corpus:
        PR[page] = 0

    # Increment value for normalization.
    inc = 1 / n

    # Choose first page at random.
    page = random.choice(list(PR.keys()))

    for _ in range(n):
        PR[page] += inc

        prob_dist = transition_model(corpus, page, damping_factor)
        page = random.choices(list(prob_dist.keys()), weights=list(prob_dist.values()))[0]
    
    return PR


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    
    # New corpus to change pages with no link.
    new_corpus = corpus.copy()
    for page in new_corpus:
        if len(new_corpus[page]) == 0:
            new_corpus[page] = set(new_corpus.keys())

    N = len(corpus)
    starting_value = 1 / N

    # Initializing 2 dicts for iteration.
    old_PR = dict()
    PR = dict()
    for page in new_corpus:
        old_PR[page] = starting_value

    # Constant factor in iterations.
    factor = (1 - damping_factor) / N

    while(True):
        repeat = False

        for page in new_corpus:
            sum = 0
            for origin_page in new_corpus:
                if page in new_corpus[origin_page]:
                    sum += old_PR[origin_page] / len(new_corpus[origin_page])

            PR[page] = factor + damping_factor * sum
            
            if abs(PR[page] - old_PR[page]) > 0.001:
                repeat = True

        if not repeat:
            break

        for page in new_corpus:
            old_PR[page] = PR[page]

    return PR


if __name__ == "__main__":
    main()
