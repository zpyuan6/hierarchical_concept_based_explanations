
def plot_results(results, random_counterpart=None, random_concepts=None, num_random_exp=100,
    min_p_val=0.05):
    """Helper function to organize results.
    When run in a notebook, outputs a matplotlib bar plot of the
    TCAV scores for all bottlenecks for each concept, replacing the
    bars with asterisks when the TCAV score is not statistically significant.
    If you ran TCAV with a random_counterpart, supply it here, otherwise supply random_concepts.
    If you get unexpected output, make sure you are using the correct keywords.
    Args:
        results: dictionary of results from TCAV runs.
        random_counterpart: name of the random_counterpart used, if it was used. 
        random_concepts: list of random experiments that were run. 
        num_random_exp: number of random experiments that were run.
        min_p_val: minimum p value for statistical significance
    """
    def is_random_concept(concept):
        if random_counterpart:
            return random_counterpart == concept
        elif random_concepts:
            return concept in random_concepts
        else:
            return 'random500_' in concept

