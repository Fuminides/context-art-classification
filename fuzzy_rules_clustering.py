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

def compute_rules_output(X, X_train_og, X_val, X_test, y):
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()

    from rpy2.robjects import pandas2ri

    names = [str(x) for x in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=names)
    X_df['y'] = y

    X_train_og_df = pd.DataFrame(X_train_og, columns=names)
    X_val_df = pd.DataFrame(X_val, columns=names)
    X_test_df = pd.DataFrame(X_test, columns=names)

    # Defining the R script and loading the instance in Python
    r = ro.r
    r['source']('fuzzy_rules.R')
    # Loading the function we have defined in R.
    filter_country_function_r = ro.globalenv['main']
    # Reading and processing data
    #converting it into r object for passing into r function
    df_r = ro.conversion.py2rpy(X_df)
    df_r_og = ro.conversion.py2rpy(X_train_og_df)
    df_val = ro.conversion.py2rpy(X_val_df)
    df_test = ro.conversion.py2rpy(X_test_df)
    #Invoking the R function and getting the result
    try:
        df_result_r = filter_country_function_r(df_r, df_r_og, df_val, df_test)

         #Converting it back to a pandas dataframe.
        consequents_train = df_result_r[0]
        consequents_train_og = df_result_r[1]
        consequents_val = df_result_r[2]
        consequents_test = df_result_r[3]
    except:
        print('Error in R')
        consequents_train = np.zeros((X_df.shape[0], 1))
        consequents_train_og = np.zeros((X_train_og_df.shape[0], 1))
        consequents_val = np.zeros((X_val_df.shape[0], 1))
        consequents_test = np.zeros((X_test_df.shape[0], 1))
   

    return consequents_train, consequents_train_og, consequents_val, consequents_test 

def frbc(X, X_val, X_test, output_clusters=128):
    m_N = X.shape[0]
    m_N_val = X_val.shape[0]
    m_N_test = X_test.shape[0]
    X_train_og = X
    y = np.array([1] * m_N, dtype=bool)
    
    q_prima = 0
    
    final_memberships = np.zeros((m_N, 128))
    final_memberships_val = np.zeros((m_N_val, 128))
    final_memberships_test = np.zeros((m_N_test, 128))

    for j in range(output_clusters):
        q = compute_q(X[y,:])
        print('Rule: ', j)
        
        while q >= q_prima:
            artificial_sample = np.random.random_sample((int(X.shape[0]*0.05), X.shape[1]))
            X = np.vstack([X, artificial_sample])
            y = np.hstack([y, np.array([0] * artificial_sample.shape[0], dtype=bool)])
            
            q_prima = compute_q(X[~y,:])
        
        # Generate rules
        consequents_train, consequents_train_og, consequents_val, consequents_test = compute_rules_output(X, X_train_og, X_val, X_test, y)

        select = consequents_train < 0.5

        final_memberships[:, j] = consequents_train_og.squeeze()
        final_memberships_val[:, j] = consequents_val.squeeze()
        final_memberships_test[:, j] = consequents_test.squeeze()

        try:
            X = X[select, :]
            y = y[select]
        except IndexError:
            return final_memberships[:, :j], final_memberships_val[:, :j], final_memberships_test[:, :j]
    
    return final_memberships, final_memberships_val, final_memberships_test

if __name__ == '__main__':
    # Load data
    top = 20

    X_train = pd.read_csv(semart_train_path).values
    X_val = pd.read_csv(semart_val_path, header=None).values
    X_test = pd.read_csv(semart_test_path, header=None).values

    arg_words = np.argsort(X_train.sum(axis=0)>0)[::-1][0:top]
    X_train = X_train[:, arg_words]
    X_val = X_val[:, arg_words]
    X_test = X_test[:, arg_words]

    # Run FRBC
    final_memberships, final_memberships_val, final_memberships_test = frbc(X_train, X_val, X_test, output_clusters=128)
    pd.DataFrame(final_memberships).to_csv('rule_embds_train.csv')
    pd.DataFrame(final_memberships_val).to_csv('rule_embds_val.csv')
    pd.DataFrame(final_memberships_test).to_csv('rule_embds_test.csv')
