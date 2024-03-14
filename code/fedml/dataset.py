import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler


def load_data():    
    train_df = pd.read_csv('survey.csv')

    # preprocessing steps
    #dealing with missing data
    train_df.drop(['comments'], axis= 1, inplace=True)
    train_df.drop(['state'], axis= 1, inplace=True)
    train_df.drop(['Timestamp'], axis= 1, inplace=True)
    
    # Assign default values for each data type
    defaultInt = 0
    defaultString = 'NaN'
    defaultFloat = 0.0

    intFeatures = ['Age']
    stringFeatures = ['Gender', 'Country', 'self_employed', 'family_history', 'treatment', 'work_interfere',
                    'no_employees', 'remote_work', 'tech_company', 'anonymity', 'leave', 'mental_health_consequence',
                    'phys_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview', 'phys_health_interview',
                    'mental_vs_physical', 'obs_consequence', 'benefits', 'care_options', 'wellness_program',
                    'seek_help']
    floatFeatures = []

    for feature in train_df:
        if feature in intFeatures:
            train_df[feature] = train_df[feature].fillna(defaultInt)
        elif feature in stringFeatures:
            train_df[feature] = train_df[feature].fillna(defaultString)
        elif feature in floatFeatures:
            train_df[feature] = train_df[feature].fillna(defaultFloat)
        else:
            print('Error: Feature %s not recognized.' % feature)
            
            
    #Made gender groups
    male_str = ["male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man","msle", "mail", "malr","cis man", "Cis Male", "cis male"]
    trans_str = ["trans-female", "something kinda male?", "queer/she/they", "non-binary","nah", "all", "enby", "fluid", "genderqueer", "androgyne", "agender", "male leaning androgynous", "guy (-ish) ^_^", "trans woman", "neuter", "female (trans)", "queer", "ostensibly male, unsure what that really means"]           
    female_str = ["cis female", "f", "female", "woman",  "femake", "female ","cis-female/femme", "female (cis)", "femail"]

    for (row, col) in train_df.iterrows():
        if str.lower(col.Gender) in male_str:
            train_df.loc[row, 'Gender'] = 'male'
        elif str.lower(col.Gender) in female_str:
            train_df.loc[row, 'Gender'] = 'female'
        elif str.lower(col.Gender) in trans_str:
            train_df.loc[row, 'Gender'] = 'trans'

    stk_list = ['A little about you', 'p']
    train_df = train_df[~train_df['Gender'].isin(stk_list)]

    # Complete missing age with median
    median_age = train_df['Age'].median()
    train_df['Age'] = train_df['Age'].fillna(median_age)


    # Fill ages < 18 and > 120 with median
    train_df.loc[train_df['Age'] < 18, 'Age'] = train_df['Age'].median()
    train_df.loc[train_df['Age'] > 120, 'Age'] = train_df['Age'].median()

    # Ranges of Age
    train_df['age_range'] = pd.cut(train_df['Age'], [0, 20, 30, 65, 100], labels=["0-20", "21-30", "31-65", "66-100"], include_lowest=True)

    train_df['self_employed'] = train_df['self_employed'].replace([defaultString], 'No')
    train_df['work_interfere'] = train_df['work_interfere'].replace([defaultString], 'Don\'t know' )
    
    # Encoding Data
    for feature in train_df:
        le = preprocessing.LabelEncoder()
        le.fit(train_df[feature])
        train_df[feature] = le.transform(train_df[feature])
        
    train_df = train_df.drop(['Country'], axis= 1)
    
    # Scaling Age
    scaler = MinMaxScaler()
    train_df['Age'] = scaler.fit_transform(train_df[['Age']])
    train_df.head()
    
    # define X and y
    feature_cols = ['Age', 'Gender', 'family_history', 'benefits', 'care_options', 'anonymity', 'leave', 'work_interfere']
    X = train_df[feature_cols]
    y = train_df.treatment

    # split X and y into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    return (X_train, y_train), (X_test, y_test)
