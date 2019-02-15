import numpy as np # Linear algebra
import pandas as pd # Data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Data visualization
import seaborn as sns # Advanced data visualization
import re # Regular expressions for advanced string selection
from mlxtend.preprocessing import OnehotTransactions # Transforming dataframe for apriori
import missingno as msno # Advanced missing values handling
from mlxtend.preprocessing import TransactionEncoder
from itertools import combinations
df = pd.read_csv('proj.csv', nrows=1000)
df.InvoiceDate = pd.to_datetime(df.InvoiceDate)
df.set_index(['InvoiceDate'] , inplace=True)

# Dropping StockCode to reduce data dimension
# Checking df.sample() for quick evaluation of entries
df.drop('StockCode', axis=1, inplace=True)
# print(df.sample(5, random_state=42))
print(df.info())
msno.bar(df);
msno.heatmap(df);

find_nans = lambda df: df[df.isnull().any(axis=1)]
dlm = 0
og_len = len(df.InvoiceNo)

def association_rules(df, metric="confidence",
                      min_threshold=0.8, support_only=False):
    """Generates a DataFrame of association rules including the
    metrics 'score', 'confidence', and 'lift'

    Parameters
    -----------
    df : pandas DataFrame
      pandas DataFrame of frequent itemsets
      with columns ['support', 'itemsets']

    metric : string (default: 'confidence')
      Metric to evaluate if a rule is of interest.
      **Automatically set to 'support' if `support_only=True`.**
      Otherwise, supported metrics are 'support', 'confidence', 'lift',
      'leverage', and 'conviction'
      These metrics are computed as follows:

      - support(A->C) = support(A+C) [aka 'support'], range: [0, 1]\n
      - confidence(A->C) = support(A+C) / support(A), range: [0, 1]\n
      - lift(A->C) = confidence(A->C) / support(C), range: [0, inf]\n
      - leverage(A->C) = support(A->C) - support(A)*support(C),
        range: [-1, 1]\n
      - conviction = [1 - support(C)] / [1 - confidence(A->C)],
        range: [0, inf]\n

    min_threshold : float (default: 0.8)
      Minimal threshold for the evaluation metric,
      via the `metric` parameter,
      to decide whether a candidate rule is of interest.

    support_only : bool (default: False)
      Only computes the rule support and fills the other
      metric columns with NaNs. This is useful if:

      a) the input DataFrame is incomplete, e.g., does
      not contain support values for all rule antecedents
      and consequents

      b) you simply want to speed up the computation because
      you don't need the other metrics.

    Returns
    ----------
    pandas DataFrame with columns "antecedents" and "consequents"
      that store itemsets, plus the scoring metric columns:
      "antecedent support", "consequent support",
      "support", "confidence", "lift",
      "leverage", "conviction"
      of all rules for which
      metric(rule) >= min_threshold.
      Each entry in the "antecedents" and "consequents" columns are
      of type `frozenset`, which is a Python built-in type that
      behaves similarly to sets except that it is immutable
      (For more info, see
      https://docs.python.org/3.6/library/stdtypes.html#frozenset).

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/

    """

    # check for mandatory columns
    if not all(col in df.columns for col in ["support", "itemsets"]):
        raise ValueError("Dataframe needs to contain the\
                         columns 'support' and 'itemsets'")

    def conviction_helper(sAC, sA, sC):
        confidence = sAC/sA
        conviction = np.empty(confidence.shape, dtype=float)
        if not len(conviction.shape):
            conviction = conviction[np.newaxis]
            confidence = confidence[np.newaxis]
            sAC = sAC[np.newaxis]
            sA = sA[np.newaxis]
            sC = sC[np.newaxis]
        conviction[:] = np.inf
        conviction[confidence < 1.] = ((1. - sC[confidence < 1.]) /
                                       (1. - confidence[confidence < 1.]))

        return conviction

    # metrics for association rules
    metric_dict = {
        "antecedent support": lambda _, sA, __: sA,
        "consequent support": lambda _, __, sC: sC,
        "support": lambda sAC, _, __: sAC,
        "confidence": lambda sAC, sA, _: sAC/sA,
        "lift": lambda sAC, sA, sC: metric_dict["confidence"](sAC, sA, sC)/sC,
        "leverage": lambda sAC, sA, sC: metric_dict["support"](
             sAC, sA, sC) - sA*sC,
        "conviction": lambda sAC, sA, sC: conviction_helper(sAC, sA, sC)
        }

    columns_ordered = ["antecedent support", "consequent support",
                       "support",
                       "confidence", "lift",
                       "leverage", "conviction"]

    # check for metric compliance
    if support_only:
        metric = 'support'
    else:
        if metric not in metric_dict.keys():
            raise ValueError("Metric must be 'confidence' or 'lift', got '{}'"
                             .format(metric))

    # get dict of {frequent itemset} -> support
    keys = df['itemsets'].values
    values = df['support'].values
    frozenset_vect = np.vectorize(lambda x: frozenset(x))
    frequent_items_dict = dict(zip(frozenset_vect(keys), values))

    # prepare buckets to collect frequent rules
    rule_antecedents = []
    rule_consequents = []
    rule_supports = []

    # iterate over all frequent itemsets
    for k in frequent_items_dict.keys():
        sAC = frequent_items_dict[k]
        # to find all possible combinations
        for idx in range(len(k)-1, 0, -1):
            # of antecedent and consequent
            for c in combinations(k, r=idx):
                antecedent = frozenset(c)
                consequent = k.difference(antecedent)

                if support_only:
                    # support doesn't need these,
                    # hence, placeholders should suffice
                    sA = None
                    sC = None

                else:
                    try:
                        sA = frequent_items_dict[antecedent]
                        sC = frequent_items_dict[consequent]
                    except KeyError as e:
                        s = (str(e) + 'You are likely getting this error'
                                      ' because the DataFrame is missing '
                                      ' antecedent and/or consequent '
                                      ' information.'
                                      ' You can try using the '
                                      ' `support_only=True` option')
                        raise KeyError(s)
                    # check for the threshold

                score = metric_dict[metric](sAC, sA, sC)
                if score >= min_threshold:
                    rule_antecedents.append(antecedent)
                    rule_consequents.append(consequent)
                    rule_supports.append([sAC, sA, sC])

    # check if frequent rule was generated
    if not rule_supports:
        return pd.DataFrame(
            columns=["antecedents", "consequents"] + columns_ordered)

    else:
        # generate metrics
        rule_supports = np.array(rule_supports).T.astype(float)
        df_res = pd.DataFrame(
            data=list(zip(rule_antecedents, rule_consequents)),
            columns=["antecedents", "consequents"])

        if support_only:
            sAC = rule_supports[0]
            for m in columns_ordered:
                df_res[m] = np.nan
            df_res['support'] = sAC

        else:
            sAC = rule_supports[0]
            sA = rule_supports[1]
            sC = rule_supports[2]
            for m in columns_ordered:
                df_res[m] = metric_dict[m](sAC, sA, sC)

        return df_res



def generate_new_combinations(old_combinations):
    """
    Generator of all combinations based on the last state of Apriori algorithm
    Parameters
    -----------
    old_combinations: np.array
        All combinations with enough support in the last step
        Combinations are represented by a matrix.
        Number of columns is equal to the combination size
        of the previous step.
        Each row represents one combination
        and contains item type ids in the ascending order
        ```
               0        1
        0      15       20
        1      15       22
        2      17       19
        ```

    Returns
    -----------
    Generator of all combinations from the last step x items
    from the previous step. Every combination is a tuple
    of item type ids in the ascending order.
    No combination other than generated
    do not have a chance to get enough support

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/generate_new_combinations/

    """

    items_types_in_previous_step = np.unique(old_combinations.flatten())
    for old_combination in old_combinations:
        max_combination = max(old_combination)
        for item in items_types_in_previous_step:
            if item > max_combination:
                res = tuple(old_combination) + (item,)
                yield res


def apriori(df, min_support=0.5, use_colnames=False, max_len=None, n_jobs=1):
    """Get frequent itemsets from a one-hot DataFrame
    Parameters
    -----------
    df : pandas DataFrame or pandas SparseDataFrame
      pandas DataFrame the encoded format.
      The allowed values are either 0/1 or True/False.
      For example,

    ```
             Apple  Bananas  Beer  Chicken  Milk  Rice
        0      1        0     1        1     0     1
        1      1        0     1        0     0     1
        2      1        0     1        0     0     0
        3      1        1     0        0     0     0
        4      0        0     1        1     1     1
        5      0        0     1        0     1     1
        6      0        0     1        0     1     0
        7      1        1     0        0     0     0
    ```

    min_support : float (default: 0.5)
      A float between 0 and 1 for minumum support of the itemsets returned.
      The support is computed as the fraction
      transactions_where_item(s)_occur / total_transactions.

    use_colnames : bool (default: False)
      If true, uses the DataFrames' column names in the returned DataFrame
      instead of column indices.

    max_len : int (default: None)
      Maximum length of the itemsets generated. If `None` (default) all
      possible itemsets lengths (under the apriori condition) are evaluated.

    Returns
    -----------
    pandas DataFrame with columns ['support', 'itemsets'] of all itemsets
      that are >= `min_support` and < than `max_len`
      (if `max_len` is not None).
      Each itemset in the 'itemsets' column is of type `frozenset`,
      which is a Python built-in type that behaves similarly to
      sets except that it is immutable
      (For more info, see
      https://docs.python.org/3.6/library/stdtypes.html#frozenset).

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/

    """
    allowed_val = {0, 1, True, False}
    unique_val = np.unique(df.values.ravel())
    for val in unique_val:
        if val not in allowed_val:
            s = ('The allowed values for a DataFrame'
                 ' are True, False, 0, 1. Found value %s' % (val))
            raise ValueError(s)

    is_sparse = hasattr(df, "to_coo")
    if is_sparse:
        X = df.to_coo().tocsc()
        support = np.array(np.sum(X, axis=0) / float(X.shape[0])).reshape(-1)
    else:
        X = df.values
        support = (np.sum(X, axis=0) / float(X.shape[0]))

    ary_col_idx = np.arange(X.shape[1])
    support_dict = {1: support[support >= min_support]}
    itemset_dict = {1: ary_col_idx[support >= min_support].reshape(-1, 1)}
    max_itemset = 1
    rows_count = float(X.shape[0])

    if max_len is None:
        max_len = float('inf')

    while max_itemset and max_itemset < max_len:
        next_max_itemset = max_itemset + 1
        combin = generate_new_combinations(itemset_dict[max_itemset])
        frequent_items = []
        frequent_items_support = []

        if is_sparse:
            all_ones = np.ones((X.shape[0], next_max_itemset))
        for c in combin:
            if is_sparse:
                together = np.all(X[:, c] == all_ones, axis=1)
            else:
                together = X[:, c].all(axis=1)
            support = together.sum() / rows_count
            if support >= min_support:
                frequent_items.append(c)
                frequent_items_support.append(support)

        if frequent_items:
            itemset_dict[next_max_itemset] = np.array(frequent_items)
            support_dict[next_max_itemset] = np.array(frequent_items_support)
            max_itemset = next_max_itemset
        else:
            max_itemset = 0

    all_res = []
    for k in sorted(itemset_dict):
        support = pd.Series(support_dict[k])
        itemsets = pd.Series([frozenset(i) for i in itemset_dict[k]])

        res = pd.concat((support, itemsets), axis=1)
        all_res.append(res)

    res_df = pd.concat(all_res)
    res_df.columns = ['support', 'itemsets']
    if use_colnames:
        mapping = {idx: item for idx, item in enumerate(df.columns)}
        res_df['itemsets'] = res_df['itemsets'].apply(lambda x: frozenset([
                                                      mapping[i] for i in x]))
    res_df = res_df.reset_index(drop=True)

    return res_df



# It does not matter not having CustomerID in this analysis
# however a NaN Description shows us a failed transaction
# We will drop NaN CustomerID when analysing customer behavior 
df.dropna(inplace=True, subset=['Description'])

# data_loss report
new_len = len(df.InvoiceNo)
dlm += (og_len - new_len)
print('Data loss report: %.2f%% of data dropped, total of %d rows' % (((og_len - new_len)/og_len), (og_len - new_len)))
print('Data loss totals: %.2f%% of total data loss, total of %d rows\n' % ((dlm/og_len), (dlm)))
mod_len = len(df.InvoiceNo)
df.info()
# Note that for dropping the rows we need the .index not a boolean list
# to_drop is a list of indices that will be used on df.drop()
to_drop = df[df.InvoiceNo.str.match('^[a-zA-Z]')].index

# Droping wrong entries starting with letters
# Our assumption is that those are devolutions or system corrections
df.drop(to_drop, axis=0, inplace=True)

# Changing data types for reducing dimension and make easier plots
df.InvoiceNo = df.InvoiceNo.astype('int64')
df.Country = df.Country.astype('category')
new_len = len(df.InvoiceNo)

# data_loss report
new_len = len(df.InvoiceNo)
dlm += (mod_len - new_len)
print('Data loss report: %.2f%% of data dropped, total of %d rows' % (((mod_len - new_len)/mod_len), (mod_len - new_len)))
print('Data loss totals: %.2f%% of total data loss, total of %d rows' % ((dlm/og_len), (dlm)))
mod_len = len(df.InvoiceNo)
country_set = df[['Country', 'InvoiceNo']]
country_set = country_set.pivot_table(columns='Country', aggfunc='count')
country_set.sort_values('InvoiceNo', axis=1, ascending=False).T



# Plotting InvoiceNo distribution per Country
plt.figure(figsize=(14,6))
plt.title('Distribuition of purchases in the website according to Countries');
sns.countplot(y='Country', data=df);


# Plotting InvoiceNo without United Kingdom
df_nUK = country_set.T.drop('United Kingdom')
plt.figure(figsize=(14,6))
plt.title('Distribuition of purchases in the website according to Countries');
# Note that since we transformed the index in type category the .remove_unused_categories is used
# otherwise it woul include a columns for United Kingdom with 0 values at the very end of the plot
sns.barplot(y=df_nUK.index.remove_unused_categories(), x='InvoiceNo', data=df_nUK, orient='h');

# Creating subsets of df for each unique country
def df_per_country(df):
    df_dict = {}
    unique_countries, counts = np.unique(df.Country, return_counts=True)
    for country in unique_countries:
        df_dict["df_{}".format(re.sub('[\s+]', '', country))] = df[df.Country == country].copy()
        # This line is giving me the warning, I will check in further research
        # After watching Data School video about the SettingWithCopyWarning I figured out the problem
        # When doing df[df.Country == country] adding the .copy() points pandas that this is an actual copy of the original df
        df_dict["df_{}".format(re.sub('[\s+]', '', country))].drop('Country', axis=1, inplace=True)
    return df_dict

# Trick to convert dictionary key/values into variables
# This way we don't need to access dfs by df_dict['df_Australia'] for example
df_dict = df_per_country(df)
locals().update(df_dict)
# Series plot function summarizing df_Countries
def series_plot(df, by1, by2, by3, period='D'):
    df_ts = df.reset_index().pivot_table(index='InvoiceDate', 
                                values=['InvoiceNo', 'Quantity', 'UnitPrice'], 
                                aggfunc=('count', 'sum'))
    df_ts = df_ts.loc[:, [('InvoiceNo', 'count'), ('Quantity', 'sum'), ('UnitPrice', 'sum')]]
    df_ts.columns = df_ts.columns.droplevel(1)
    plt.figure(figsize=(14, 6))
    
    plt.subplot(2, 2, 1)
    plt.plot(df_ts.resample(period).sum().bfill()[[by1]], color='navy')
    plt.title('{}'.format(by1));
    plt.xticks(rotation=60);
    plt.subplot(2, 2, 2)
    plt.title('{}'.format(by2));
    plt.plot(df_ts.resample(period).sum().bfill()[[by2]], label='Total Sale', color='orange');
    plt.xticks(rotation=60)
    plt.tight_layout()
    
    plt.figure(figsize=(14, 8))
    plt.title('{}'.format(by3));
    plt.plot(df_ts.resample(period).sum().bfill()[[by3]], label='Total Invoices', color='green');
    plt.tight_layout()
    #plt.show()



series_plot(df_UnitedKingdom, 'Quantity', 'UnitPrice', 'InvoiceNo')

# Starting preparation of df for receiving product association
# Cleaning Description field for proper aggregation 
df_UnitedKingdom.loc[:, 'Description'] = df_UnitedKingdom.Description.str.strip()
# Once again, this line was generating me the SettingWithCopyWarning, solved by adding the .copy()

# Dummy conding and creation of the baskets_sets, indexed by InvoiceNo with 1 corresponding to every item presented on the basket
# Note that the quantity bought is not considered, only if the item was present or not in the basket
basket = pd.get_dummies(df_UnitedKingdom.reset_index().loc[:, ('InvoiceNo', 'Description')])

basket_sets = pd.pivot_table(basket, index='InvoiceNo', aggfunc='sum')
print(type(basket_sets))
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_sets = basket_sets.applymap(encode_units)
#basket_sets.drop('POSTAGE', inplace=True, axis=1)
# Apriori aplication: frequent_itemsets
# Note that min_support parameter was set to a very low value, this is the Spurious limitation, more on conclusion section
frequent_itemsets = apriori(basket_sets, min_support=0.03, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

# Advanced and strategical data frequent set selection
frequent_itemsets[ (frequent_itemsets['length'] > 1) &
                   (frequent_itemsets['support'] >= 0.02) ].head()
print(frequent_itemsets)
# Generating the association_rules: rules
# Selecting the important parameters for analysis
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

rules[['antecedants', 'consequents', 'support', 'confidence', 'lift']].sort_values('support', ascending=False).head()

# Visualizing the rules distribution color mapped by Lift
plt.figure(figsize=(14, 8))
plt.scatter(rules_slice['support'], rules_slice['confidence'], c=rules_slice['lift'], alpha=0.9, cmap='YlOrRd');
plt.title('Rules distribution color mapped by lift');
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.colorbar();
plt.show();
