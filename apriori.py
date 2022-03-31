# This is Modify version from Apriori algorithm by Ali Yavari For Profile-based Fuzzy Association Rule Mining Algorithm
# Original Code Written by, Sebastian Raschka <sebastianraschka.com> and Available in Github:   http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/generate_new_combinations/

import numpy as np
import pandas as pd


def generate_new_combinations(old_combinations,types):
    items_types_in_previous_step = np.unique(old_combinations.flatten())
    types_ = types[items_types_in_previous_step]
    for old_combination in old_combinations:
        max_combination = old_combination[-1]
        mask = (items_types_in_previous_step > max_combination) 
        mask_ = types_[max_combination] != types_[0:]
        mask = mask & mask_
        valid_items = items_types_in_previous_step[mask]
        old_tuple = tuple(old_combination)
        for item in valid_items:
            yield from old_tuple
            yield item


def generate_new_combinations_low_memory(old_combinations, X, min_support,
                                         is_sparse, types):
    items_types_in_previous_step = np.unique(old_combinations.flatten())
    types_ = types[items_types_in_previous_step]
    rows_count = X.shape[0]
    threshold = min_support * rows_count
    for old_combination in old_combinations:
        max_combination = old_combination[-1]
        mask = items_types_in_previous_step > max_combination
        mask_ = types_[max_combination] != types_[0:]
        mask = mask & mask_
        valid_items = items_types_in_previous_step[mask]
        old_tuple = tuple(old_combination)
        if is_sparse:
            mask_rows = X[:, old_tuple].toarray().all(axis=1)
            X_cols = X[:, valid_items].toarray()
            supports = X_cols[mask_rows].sum(axis=0)
        else:
            
            lhs = np.min(X[:, old_tuple],axis=1)
            lhs = lhs.reshape(len(lhs),1)
            supports = np.minimum(lhs,X[:, valid_items]).sum(axis=0)
            
            
        valid_indices = (supports >= threshold).nonzero()[0]
        for index in valid_indices:
            yield float(supports[index])
            yield from old_tuple
            yield valid_items[index]


def apriori(df,types, min_support=0.5, use_colnames=False, max_len=None, verbose=0, low_memory=False):


    def _support(_x, _n_rows, _is_sparse):        
        out = (np.sum(_x, axis=0) / _n_rows)
        return np.array(out).reshape(-1)

    if min_support <= 0.:
        raise ValueError('`min_support` must be a positive '
                         'number within the interval `(0, 1]`. '
                         'Got %s.' % min_support)

    #fpc.valid_input_check(df)

    if hasattr(df, "sparse"):
        # DataFrame with SparseArray (pandas >= 0.24)
        if df.size == 0:
            X = df.values
        else:
            X = df.sparse.to_coo().tocsc()
        is_sparse = True
    else:
        # dense DataFrame
        X = df.values
        is_sparse = False
    support = _support(X, X.shape[0], is_sparse)
    ary_col_idx = np.arange(X.shape[1])
    support_dict = {1: support[support >= min_support]}
    itemset_dict = {1: ary_col_idx[support >= min_support].reshape(-1, 1)}
    types = types[support >= min_support]
    max_itemset = 1
    rows_count = float(X.shape[0])

    all_ones = np.ones((int(rows_count), 1))

    while max_itemset and max_itemset < (max_len or float('inf')):
        next_max_itemset = max_itemset + 1

        
        if low_memory:            
            combin = generate_new_combinations_low_memory(itemset_dict[max_itemset], X, min_support, is_sparse,types)
            
            # slightly faster than creating an array from a list of tuples
            combin = np.fromiter(combin, dtype=(float))
            combin = combin.reshape(-1, next_max_itemset + 1)

            if combin.size == 0:
                break
            if verbose:
                print(
                    '\rProcessing %d combinations | Sampling itemset size %d' %
                    (combin.size, next_max_itemset), end="")

            itemset_dict[next_max_itemset] = combin[:, 1:].astype(int)
            support_dict[next_max_itemset] = combin[:, 0].astype(float) \
                / rows_count
            max_itemset = next_max_itemset
        else:
            combin = generate_new_combinations(itemset_dict[max_itemset],types)
            combin = np.fromiter(combin, dtype=int)
            combin = combin.reshape(-1, next_max_itemset)

            if combin.size == 0:
                break
            if verbose:
                print(
                    '\rProcessing %d combinations | Sampling itemset size %d' %
                    (combin.size, next_max_itemset), end="")

            if is_sparse:
                bools = X[:, combin[:, 0]] == all_ones
                for n in range(1, combin.shape[1]):
                    bools = bools & (X[:, combin[:, n]] == all_ones)
            else:
                sups_ = np.min(X[:, combin],axis=2)

            support = _support(np.array(sups_), rows_count, is_sparse)
            _mask = (support >= min_support).reshape(-1)
            if any(_mask):
                itemset_dict[next_max_itemset] = np.array(combin[_mask])
                support_dict[next_max_itemset] = np.array(support[_mask])
                max_itemset = next_max_itemset                
            else:
                # Exit condition
                break

    all_res = []
    for k in sorted(itemset_dict):
        support = pd.Series(support_dict[k])
        itemsets = pd.Series([frozenset(i) for i in itemset_dict[k]],
                             dtype='object')

        res = pd.concat((support, itemsets), axis=1)
        all_res.append(res)

    res_df = pd.concat(all_res)
    res_df.columns = ['support', 'itemsets']
    if use_colnames:
        mapping = {idx: item for idx, item in enumerate(df.columns)}
        res_df['itemsets'] = res_df['itemsets'].apply(lambda x: frozenset([
                                                      mapping[i] for i in x]))
    res_df = res_df.reset_index(drop=True)

    if verbose:
        print()  # adds newline if verbose counter was used

    return res_df
