#import graphviz as graphviz
from sklearn.datasets import load_iris
from sklearn import tree
import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
data = pd.read_csv('data_trial.csv')


X = data.drop('Risk', axis=1)
y = data['Risk']

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=45, test_size=.2)

st.title("Welcome to our simple machine learning app")
st.subheader("Please setting these kind if you want to see the trees.")
st.write(data.head())


max_dep = st.selectbox('How many maximum depth you want to?',([x for x in range(1,6)]))
min_spl = st.slider("How many minimum sample split you want to?", 1,50,1)
min_lea = st.slider("How many minimum sample leaf you want to?", 1,50,1)

tree_model = DecisionTreeClassifier(criterion='entropy', max_depth = max_dep, min_samples_split = min_spl,
                                                        min_samples_leaf=min_lea)

if st.button("Show the trees!"):
    clf = tree_model.fit(X_train, y_train)

    dot_data  = tree.export_graphviz(clf,feature_names=X.columns,class_names=['Bad',"Good"], out_file=None)

    st.subheader("Your model accuracy is {}".format(clf.score(X_test, y_test)))

    st.graphviz_chart(dot_data)