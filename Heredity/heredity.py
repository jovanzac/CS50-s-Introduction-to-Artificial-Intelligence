import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def person_gene(person, one_gene, two_genes) :
    if person in one_gene : return 1
    elif person in two_genes : return 2
    else : return 0


def gene_inheritance_prob(gene_num, inherited=False) :
    if gene_num == 0 :
        if inherited :
            return PROBS["mutation"]
        else :
            return 1 - PROBS["mutation"]
    elif gene_num == 1 :
        return 0.5
    elif gene_num == 2 :
        if inherited :
            return 1 - PROBS["mutation"]
        else :
            return PROBS["mutation"]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    joint_prob = 1
    for person in people :
        gene_num = person_gene(person, one_gene, two_genes)
        if people[person]["mother"] is None and people[person]["father"] is None :
            gene_prob = PROBS["gene"][gene_num]
            if person in have_trait :
                trait_prob = PROBS["trait"][gene_num][True]
            else :
                trait_prob = PROBS["trait"][gene_num][False]
            joint_prob *= gene_prob*trait_prob
        else :
            mom_gene = person_gene(people[person]["mother"], one_gene, two_genes)
            dad_gene = person_gene(people[person]["father"], one_gene, two_genes)
            if person in two_genes :
                probability = gene_inheritance_prob(mom_gene, True) * gene_inheritance_prob(dad_gene, True)
            elif person in one_gene :
                probability = gene_inheritance_prob(mom_gene, True)*gene_inheritance_prob(dad_gene, False) + gene_inheritance_prob(dad_gene,True)*gene_inheritance_prob(mom_gene, False)
            else :
                probability = gene_inheritance_prob(mom_gene, False) * gene_inheritance_prob(dad_gene, False)
            joint_prob *= probability * PROBS["trait"][gene_num][person in have_trait]
    return joint_prob


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person in probabilities :
        gene_num = person_gene(person, one_gene, two_genes)
        probabilities[person]["gene"][gene_num] += p
        probabilities[person]["trait"][person in have_trait] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for person in probabilities :
        gene_raw, trait_raw = list(), list()
        for gene_num in probabilities[person]["gene"] :
            gene_raw += [probabilities[person]["gene"][gene_num],]
        for inherited in probabilities[person]["trait"] :
            trait_raw += [probabilities[person]["trait"][inherited],]
        # normalizing gene values
        gene_normalized = [value/sum(gene_raw) for value in gene_raw]
        # normalizing trait values
        trait_normalized = [value/sum(trait_raw) for value in trait_raw]
        for inherited in [False, True] :
            probabilities[person]["trait"][inherited] = trait_normalized.pop()
        for gene_num in range(3) :
            probabilities[person]["gene"][gene_num] = gene_normalized.pop()


if __name__ == "__main__":
    main()
