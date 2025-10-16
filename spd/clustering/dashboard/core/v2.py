from dataclasses import dataclass


@dataclass
class ClusterData:
    # cluster id
    # list of subcomponent keys
    # for future reference: "ckeys" means this_cluster_id_hash|subcomponent_key
    # activations of subcomponents on all text : dict[ckeys, dict[text_hash, activation array]]

    # properties: these are stats we compute and care about. these will be kept as-is

    # max activating sequences: dict[ckeys: list of text_hash]
    # whatever other activating sequences -- i.e. maximum mean, most above threshold, etc.

    # subcomponent coactivations
    # various activation stats (across all sequences)
    # top activating tokens
    pass


# now, when we store the data we care about -- call this class DashboardData, we will take in a bunch of ClusterData and put things into shared data structures.
# i.e. we remove activations of components/subcomponents on all text from ClusterData, and instead store a big dict[(ckeys, text_hash), activation array]
# we store text data in a big dict[text_hash]: text data
# we store subcomponent coactivations in a big dict[cluster_id, array] (saved as npz)

# BUT -- most importantly, we want this concatenation to basically be as automated as possible. we should be able to add fields freely to cluster data, and then somehow mark them as concatenated. then we want this all to be super easy to read from javascript
