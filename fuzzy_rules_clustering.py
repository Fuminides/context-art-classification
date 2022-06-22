semart_train_path = ''
semart_val_path = ''
semart_test_path = ''

import numpy as np
import pandas as pd


def compute_q(X):
    centroid = np.mean(X) #I interpreted right? God knows
    return np.sum((X - centroid)**2, axis=-1)

def best_rule(rules_antecedents):
    return np.argmax(np.prod(rules_antecedents, axis=1))

def compute_rules_output(X, y):
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri

    names = [str(x) for x in range(X.shape[0])]
    X_df = pd.DataFrame(X, columns=names)
    X_df['y'] = y

    # Defining the R script and loading the instance in Python
    r = robjects.r
    r['source']('fuzzy_rules.R')
    # Loading the function we have defined in R.
    filter_country_function_r = robjects.globalenv['main']
    # Reading and processing data
    #converting it into r object for passing into r function
    df_r = pandas2ri.ri2py(X_df)
    #Invoking the R function and getting the result
    df_result_r = filter_country_function_r(df_r)
    #Converting it back to a pandas dataframe.
    df_result = pandas2ri.py2ri(df_result_r)

    antecedents = df_result.iloc[:, :-1].values
    consequents = df_result.iloc[:, -1].values

    return antecedents, consequents

def frbc(X, output_clusters=128):
    m_N = X.shape[0]
    y = [0] * m_N
    
    q = 0
    q_prima = 0
    
    final_memberships = np.zeros(m_N, 128)

    for j in range(output_clusters):
        artificial_samples = []
        while q >= q_prima:
            artificial_sample = np.random.random_sample((X.shape[1]))
            artificial_samples.append(artificial_sample)

            X = np.vstack([X, artificial_sample])
            y.append(1)

            q = compute_q(X)
            q_prima = compute_q(artificial_samples)
        
        # Generate rules
        antecedents, consequents = compute_rules_output(X, np.array(y))

        best_rule = best_rule(antecedents)
        select = consequents < 0.5

        final_memberships[j, :] = consequents[:, best_rule]
        X = X[select, :]
        y = [z for ix, z in enumerate(y) if select[ix]]
    
    return final_memberships

