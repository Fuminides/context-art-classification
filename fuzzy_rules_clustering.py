semart_train_path = 'tfidf_semart_train.csv'
semart_val_path = 'tfidf_semart_val.csv'
semart_test_path = 'tfidf_semart_test.csv'

import numpy as np
import pandas as pd


def compute_q(X):
    centroid = np.mean(X, axis=0) #I interpreted right? God knows
    return np.sum((X - centroid)**2, axis=None)

def best_rule(rules_antecedents):
    return np.argmax(np.prod(rules_antecedents, axis=1))

def compute_rules_output(X, X_val, X_test, y):
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri

    names = [str(x) for x in range(X.shape[0])]
    X_df = pd.DataFrame(X, columns=names)
    X_df['y'] = y

    X_val_df = pd.DataFrame(X_val, columns=names)
    X_val_df['y'] = y

    X_test_df = pd.DataFrame(X_test, columns=names)
    X_test_df['y'] = y

    # Defining the R script and loading the instance in Python
    r = robjects.r
    r['source']('fuzzy_rules.R')
    # Loading the function we have defined in R.
    filter_country_function_r = robjects.globalenv['main']
    # Reading and processing data
    #converting it into r object for passing into r function
    df_r = pandas2ri.ri2py(X_df)
    #Invoking the R function and getting the result
    df_result_r = filter_country_function_r(df_r, X_val_df, X_test_df)
    #Converting it back to a pandas dataframe.
    df_result = pandas2ri.py2ri(df_result_r)

    consequents_train = df_result.iloc[:, -3].values
    consequents_val = df_result.iloc[:, -2].values
    consequents_test = df_result.iloc[:, -1].values

    return consequents_train, consequents_val, consequents_test 

def frbc(X, X_val, X_test, output_clusters=128):
    m_N = X.shape[0]
    m_N_val = X_val.shape[0]
    m_N_test = X_test.shape[0]

    y = [0] * m_N
    
    q = 0
    q_prima = 0
    
    final_memberships = np.zeros((m_N, 128))
    final_memberships_val = np.zeros((m_N_val, 128))
    final_memberships_test = np.zeros((m_N_test, 128))

    for j in range(output_clusters):
        print('Rule: ', j)
        artificial_samples = []
        while q >= q_prima:
            artificial_sample = np.random.random_sample((X.shape[1]))
            artificial_samples.append(artificial_sample)

            X = np.vstack([X, artificial_sample])
            y.append(1)

            q = compute_q(X)
            q_prima = compute_q(np.array(artificial_samples))
        
        # Generate rules
        consequents_train, consequents_val, consequents_test = compute_rules_output(X, np.array(y))

        select = consequents_train < 0.5

        final_memberships[j, :] = consequents_train
        final_memberships_val[j, :] = consequents_val
        consequents_test[j, :] = consequents_test
        X = X[select, :]
        y = [z for ix, z in enumerate(y) if select[ix]]
    
    return final_memberships, final_memberships_val, final_memberships_test

if __name__ == '__main__':
    # Load data
    X_train = pd.read_csv(semart_train_path).values
    X_val = pd.read_csv(semart_val_path).values
    X_test = pd.read_csv(semart_test_path).values

    # Run FRBC
    final_memberships, final_memberships_val, final_memberships_test = frbc(X_train, X_val, X_test, output_clusters=128)
    final_memberships.to_csv('rule_embds_train.csv')
    final_memberships_val.to_csv('rule_embds_val.csv')
    final_memberships_test.to_csv('rule_embds_test.csv')
