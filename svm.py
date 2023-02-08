from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix

clf = svm.SVC(kernel='sigmoid', gamma=0.5, C=32).fit(X_train, y_train) # 高斯kernel
clf.fit(X_train,y_train)
predict_list = clf.predict(X_test)
precition = clf.score(X_test,y_test)
    
print('precision is : ',precition*100,"%")
cm = confusion_matrix(y_test, predict_list)clf = svm.SVC(kernel='sigmoid', gamma=0.5, C=32).fit(X_train, y_train) # 高斯kernel
clf.fit(X_train,y_train)
predict_list = clf.predict(X_test)
precition = clf.score(X_test,y_test)
    
print('precision is : ',precition*100,"%")
cm = confusion_matrix(y_test, predict_list)
