# WEEK 3: Unsupervised Learning & Model Evaluation
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Load cleaned data (after Week 1 cleaning)
df = pd.read_csv('train.csv') # use your cleaned file
X = df[['Pclass','Sex','Age','Fare','SibSp','Parch']].fillna(df['Age'].mean())
X['Sex'] = X['Sex'].map({'male':0,'female':1})

# Scale for clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 1. K-Means + Elbow Method
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss); plt.title('Elbow Method'); plt.show()

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# 2. Hierarchical Clustering
link = linkage(X_scaled, method='ward')
dendrogram(link)
plt.show()
agg = AgglomerativeClustering(n_clusters=3)
agg_clusters = agg.fit_predict(X_scaled)

# 3. PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters); plt.show()

# 4. Model Evaluation (using Week 2 model)
y = df['Survived']
rf = RandomForestClassifier(random_state=42)

# Cross-validation
cv_scores = cross_val_score(rf, X_scaled, y, cv=5)
print("CV Mean:", cv_scores.mean())

# Confusion Matrix, Precision, Recall, F1, ROC-AUC
rf.fit(X_scaled, y)
# (for demo use train_test_split in real)
from sklearn.model_selection import train_test_split
Xtr, Xte, ytr, yte = train_test_split(X_scaled, y, test_size=0.2)
rf.fit(Xtr, ytr)
pred = rf.predict(Xte)
print(confusion_matrix(yte, pred))
print(classification_report(yte, pred))
print("ROC-AUC:", roc_auc_score(yte, rf.predict_proba(Xte)[:,1]))

# 5. Hyperparameter Tuning
grid = GridSearchCV(rf, {'n_estimators':[50,100,150], 'max_depth':[5,10,None]}, cv=3)
grid.fit(Xtr, ytr)
print("Best Params:", grid.best_params_, grid.best_score_)
