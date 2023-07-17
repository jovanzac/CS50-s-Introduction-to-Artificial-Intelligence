import os
import random
import re
import sys
import math

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
    model = dict()
    page_count = len(corpus)
    # if there are no pages linked to current page return equal probability for each page
    if len(corpus[page]) == 0 :
        for i in corpus :
            model[i] = 1/page_count

        return model
    # otherwise return respective probabilities
    for i in corpus :
        model[i] = (1-damping_factor)/page_count
        if i in corpus[page] :
            model[i] += damping_factor/len(corpus[page])

    return model


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pagerank = dict()
    page = random.choice(list(corpus))
    for _ in range(n-1) :
        model = transition_model(corpus, page, damping_factor)
        page = random.choices(list(model), weights=model.values(), k=1).pop()
        if page in pagerank :
            pagerank[page] += 1
        else :
            pagerank[page] = 1
    for i in pagerank :
        pagerank[i] = pagerank[i]/n
    
    return pagerank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    total_pages = len(corpus)
    pagerank = dict(zip(corpus.keys(), [1/total_pages] * total_pages))
    pagerank_diff = dict(zip(corpus.keys(), [math.inf] * total_pages))
    while any(rank_diff > 0.001 for rank_diff in pagerank_diff.values()) :
        for page in pagerank.keys() :
            probability = 0
            for link_page, links in corpus.items() :
                if not links :
                    links = corpus.keys()
                if page in links :
                    probability += pagerank[link_page]/len(links)
            new_pagerank = (1-damping_factor)/total_pages + (damping_factor*probability)
            pagerank_diff[page] = abs(new_pagerank - pagerank[page])
            pagerank[page] = new_pagerank
    return pagerank

if __name__ == "__main__":
    main()
