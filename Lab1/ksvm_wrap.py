from sklearn import svm
class KSVMWrap:
    def __init__(self, X, Y_, c=1, g='auto'):
        """Arguments:
            X,Y_: podatci i točni indeksi razreda
            c:    relativni značaj podatkovne cijene
            g:    širina RBF jezgre
        """
        self.clf = svm.SVC(C=c,gamma=g,probability=True)
        self.clf.fit(X, Y_)

    def predict(self, X):
        return self.clf.fit(X, Y_)
        
    def scores(self, X):
        #np.where(clf.classes_ == zeljena klasa)
        return self.clf.classes_, self.clf.predict_proba(X) 
    
    def support(self):
        return self.clf.support_