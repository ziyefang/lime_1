
# coding: utf-8

# In[1]:
from __future__ import print_function
import sklearn
import sklearn.datasets
import sklearn.ensemble
import numpy as np
import lime
import lime.lime_tabular

np.random.seed(1)


# ## Continuous features

# ### Loading data, training a model

# For this part, we'll use the Iris dataset, and we'll train a random forest. 

# In[2]:

iris = sklearn.datasets.load_iris()


# In[3]:

train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(iris.data, iris.target, train_size=0.80)


# In[4]:

rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
rf.fit(train, labels_train)


# In[5]:

sklearn.metrics.accuracy_score(labels_test, rf.predict(test))


# ### Create the explainer

# As opposed to lime_text.TextExplainer, tabular explainers need a training set. The reason for this is because we compute statistics on each feature (column). If the feature is numerical, we compute the mean and std, and discretize it into quartiles. If the feature is categorical, we compute the frequency of each value. For this tutorial, we'll only look at numerical features.

# We use these computed statistics for two things:
# 1. To scale the data, so that we can meaningfully compute distances when the attributes are not on the same scale
# 2. To sample perturbed instances - which we do by sampling from a Normal(0,1), multiplying by the std and adding back the mean.
# 

# In[6]:

explainer = lime.lime_tabular.LimeTabularExplainer(train, feature_names=iris.feature_names, class_names=iris.target_names, discretize_continuous=True)


# ### Explaining an instance

# Since this is a multi-class classification problem, we set the top_labels parameter, so that we only explain the top class.

# In[7]:

i = np.random.randint(0, test.shape[0])
exp = explainer.explain_instance(test[i], rf.predict_proba, num_features=2, top_labels=1)


# We now explain a single instance:

# In[8]:

exp.show_in_notebook(show_table=True, show_all=False)


# Now, there is a lot going on here. First, note that the row we are explained is displayed on the right side, in table format. Since we had the show_all parameter set to false, only the features used in the explanation are displayed.
# 
# The *value* column displays the original value for each feature.
# 
# Note that LIME has discretized the features in the explanation. This is because we let discretize_continuous=True in the constructor (this is the default). Discretized features make for more intuitive explanations.
# 

# ### Checking the local linear approximation

# In[9]:

feature_index = lambda x: iris.feature_names.index(x)


# In[10]:

print('Increasing petal width')
temp = test[i].copy()
print('P(setosa) before:', rf.predict_proba(temp.reshape(1,-1))[0,0])
temp[feature_index('petal width (cm)')] = 1.5
print('P(setosa) after:', rf.predict_proba(temp.reshape(1,-1))[0,0])
print ()
print('Increasing petal length')
temp = test[i].copy()
print('P(setosa) before:', rf.predict_proba(temp.reshape(1,-1))[0,0])
temp[feature_index('petal length (cm)')] = 3.5
print('P(setosa) after:', rf.predict_proba(temp.reshape(1,-1))[0,0])
print()
print('Increasing both')
temp = test[i].copy()
print('P(setosa) before:', rf.predict_proba(temp.reshape(1,-1))[0,0])
temp[feature_index('petal width (cm)')] = 1.5
temp[feature_index('petal length (cm)')] = 3.5
print('P(setosa) after:', rf.predict_proba(temp.reshape(1,-1))[0,0])


# Note that both features had the impact we thought they would. The scale at which they need to be perturbed of course depends on the scale of the feature in the training set.

# We now show all features, just for completeness:

# In[11]:

exp.show_in_notebook(show_table=True, show_all=True)


# ## Categorical features

# For this part, we will use the Mushroom dataset, which will be downloaded [here](http://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data). The task is to predict if a mushroom is edible or poisonous, based on categorical features.

# ### Loading data

# In[12]:

data = np.genfromtxt('http://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data', delimiter=',', dtype='<U20')
labels = data[:,0]
le= sklearn.preprocessing.LabelEncoder()
le.fit(labels)
labels = le.transform(labels)
class_names = le.classes_
data = data[:,1:]


# In[13]:

categorical_features = range(22)


# In[14]:

feature_names = 'cap-shape,cap-surface,cap-color,bruises?,odor,gill-attachment,gill-spacing,gill-size,gill-color,stalk-shape,stalk-root,stalk-surface-above-ring, stalk-surface-below-ring, stalk-color-above-ring,stalk-color-below-ring,veil-type,veil-color,ring-number,ring-type,spore-print-color,population,habitat'.split(',')


# We expand the characters into words, using the data available in the UCI repository

# In[15]:

categorical_names = '''bell=b,conical=c,convex=x,flat=f,knobbed=k,sunken=s
fibrous=f,grooves=g,scaly=y,smooth=s
brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y
bruises=t,no=f
almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s
attached=a,descending=d,free=f,notched=n
close=c,crowded=w,distant=d
broad=b,narrow=n
black=k,brown=n,buff=b,chocolate=h,gray=g,green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y
enlarging=e,tapering=t
bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?
fibrous=f,scaly=y,silky=k,smooth=s
fibrous=f,scaly=y,silky=k,smooth=s
brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
partial=p,universal=u
brown=n,orange=o,white=w,yellow=y
none=n,one=o,two=t
cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z
black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y
abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y
grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d'''.split('\n')
for j, names in enumerate(categorical_names):
    values = names.split(',')
    values = dict([(x.split('=')[1], x.split('=')[0]) for x in values])
    data[:,j] = np.array(list(map(lambda x: values[x], data[:,j])))


# Our explainer (and most classifiers) takes in numerical data, even if the features are categorical. We thus transform all of the string attributes into integers, using sklearn's LabelEncoder. We use a dict to save the correspondence between the integer values and the original strings, so that we can present this later in the explanations.

# In[16]:

categorical_names = {}
for feature in categorical_features:
    le = sklearn.preprocessing.LabelEncoder()
    le.fit(data[:, feature])
    data[:, feature] = le.transform(data[:, feature])
    categorical_names[feature] = le.classes_


# In[17]:

data[:,0]


# In[18]:

categorical_names[0]


# We now split the data into training and testing

# In[19]:

data = data.astype(float)


# In[20]:

train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(data, labels, train_size=0.80)


# Finally, we use a One-hot encoder, so that the classifier does not take our categorical features as continuous features. We will use this encoder only for the classifier, not for the explainer - and the reason is that the explainer must make sure that a categorical feature only has one value.

# In[21]:

encoder = sklearn.preprocessing.OneHotEncoder(categorical_features=categorical_features)


# In[22]:

encoder.fit(data)
encoded_train = encoder.transform(train)


# In[23]:

rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
rf.fit(encoded_train, labels_train)


# Note that our predict function first transforms the data into the one-hot representation

# In[24]:

predict_fn = lambda x: rf.predict_proba(encoder.transform(x))


# This classifier has perfect accuracy on the test set!

# In[25]:

sklearn.metrics.accuracy_score(labels_test, rf.predict(encoder.transform(test)))


# ### Explaining predictions

# We now create our explainer. The categorical_features parameter lets it know which features are categorical (in this case, all of them). The categorical names parameter gives a string representation of each categorical feature's numerical value, as we saw before.

# In[26]:

np.random.seed(1)


# In[27]:

explainer = lime.lime_tabular.LimeTabularExplainer(train ,class_names=['edible', 'poisonous'], feature_names = feature_names,
                                                   categorical_features=categorical_features, 
                                                   categorical_names=categorical_names, kernel_width=3, verbose=False)


# In[28]:

i = 137
exp = explainer.explain_instance(test[i], predict_fn, num_features=5)
exp.show_in_notebook()


# Now note that the explanations are based not only on features, but on feature-value pairs. For example, we are saying that odor=foul is indicative of a poisonous mushroom. In the context of a categorical feature, odor could take many other values (see below). Since we perturb each categorical feature drawing samples according to the original training distribution, the way to interpret this is: if odor was not foul, on average, this prediction would be 0.24 less 'poisonous'. Let's check if this is the case

# In[29]:

odor_idx = feature_names.index('odor')
explainer.categorical_names[odor_idx]


# In[30]:

explainer.feature_frequencies[odor_idx]


# In[31]:

foul_idx = 4
non_foul = np.delete(explainer.categorical_names[odor_idx], foul_idx)
non_foul_normalized_frequencies = explainer.feature_frequencies[odor_idx].copy()
non_foul_normalized_frequencies[foul_idx] = 0
non_foul_normalized_frequencies /= non_foul_normalized_frequencies.sum()


# In[32]:

print('Making odor not equal foul')
temp = test[i].copy()
print('P(poisonous) before:', predict_fn(temp.reshape(1,-1))[0,1])
print
average_poisonous = 0
for idx, (name, frequency) in enumerate(zip(explainer.categorical_names[odor_idx], non_foul_normalized_frequencies)):
    if name == 'foul':
        continue
    temp[odor_idx] = idx
    p_poisonous = predict_fn(temp.reshape(1,-1))[0,1]
    average_poisonous += p_poisonous * frequency
    print('P(poisonous | odor=%s): %.2f' % (name, p_poisonous))
print ()
print ('P(poisonous | odor != foul) = %.2f' % average_poisonous)


# We see that in this particular case, the linear model is pretty close: it predicted that on average odor increases the probability of poisonous by 0.26, when in fact it is by 0.23. Notice though that we only changed one feature (odor), when the linear model takes into account perturbations of all the features at once.

# ## Numerical and Categorical features in the same dataset
# 

# We now turn to a dataset that has both numerical and categorical features. Here, the task is to predict whether a person makes over 50K dollars per year. Downloads the data [here](http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data).

# In[33]:

feature_names = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status","Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss","Hours per week", "Country"]


# In[34]:

data = np.genfromtxt('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', delimiter=', ', dtype=str)


# In[35]:

labels = data[:,14]
le= sklearn.preprocessing.LabelEncoder()
le.fit(labels)
labels = le.transform(labels)
class_names = le.classes_
data = data[:,:-1]


# In[36]:

categorical_features = [1,3,5, 6,7,8,9,13]


# In[37]:

categorical_names = {}
for feature in categorical_features:
    le = sklearn.preprocessing.LabelEncoder()
    le.fit(data[:, feature])
    data[:, feature] = le.transform(data[:, feature])
    categorical_names[feature] = le.classes_


# In[38]:

data = data.astype(float)


# In[39]:

encoder = sklearn.preprocessing.OneHotEncoder(categorical_features=categorical_features)


# In[40]:

np.random.seed(1)
train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(data, labels, train_size=0.80)


# In[41]:

encoder.fit(data)
encoded_train = encoder.transform(train)


# This time, we use gradient boosted trees as the model, using the [xgboost](https://github.com/dmlc/xgboost) package.

# In[42]:

import xgboost
gbtree = xgboost.XGBClassifier(n_estimators=300, max_depth=5)
gbtree.fit(encoded_train, labels_train)


# In[43]:

sklearn.metrics.accuracy_score(labels_test, gbtree.predict(encoder.transform(test)))


# In[44]:

predict_fn = lambda x: gbtree.predict_proba(encoder.transform(x)).astype(float)


# ### Explaining predictions

# In[45]:

explainer = lime.lime_tabular.LimeTabularExplainer(train ,feature_names = feature_names,class_names=class_names,
                                                   categorical_features=categorical_features, 
                                                   categorical_names=categorical_names, kernel_width=3)


# We now show a few explanations. These are just a mix of the continuous and categorical examples we showed before. For categorical features, the feature contribution is always the same as the linear model weight.

# In[46]:

np.random.seed(1)
i = 1653
exp = explainer.explain_instance(test[i], predict_fn, num_features=5)
exp.show_in_notebook(show_all=False)


# Note that capital gain has very high weight. This makes sense. Now let's see an example where the person has a capital gain below the mean:

# In[47]:

i = 10
exp = explainer.explain_instance(test[i], predict_fn, num_features=5)
exp.show_in_notebook(show_all=False)


# In[ ]:



